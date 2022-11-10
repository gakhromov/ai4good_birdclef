import os
import pandas as pd
from config import config, signal_conf
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
import pathlib
from helpers import *
import torch


def combine_labels(prim, sec):
    # remove all the signs
    sec = sec.replace("'", "")
    sec = sec.replace("[", "")
    sec = sec.replace("]", "")
    sec = sec.replace(" ", "")

    sec_items = sec.split(',')
    if sec_items[0] == '':
        return [prim]
    comb = [prim]
    comb = comb + sec_items
    return comb


def get_dataset(args):
    path = args.data_path
    secondary = args.use_secondary

    print(f"Read config {config}")
    df = pd.read_csv(f'{path}/augmented.csv')

    # make folds, stratify with primary labels anyways, even when we have secondary label usage
    skf = StratifiedGroupKFold(n_splits=config['n_folds'], shuffle=True)
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label'], groups=df['file'])):
        df.loc[val_ind, 'fold'] = k

    # generate n_fold datasets
    folded_ds = []
    for fold in range(config['n_folds']):
        train_ds, valid_ds = get_data(df, fold, args, sec=secondary)
        folded_ds.append((train_ds, valid_ds))

    return folded_ds


def get_data(df, fold, args, type="mel", sec=True):
    # extract fold
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    # if ogg, load the data from ogg files, convert to numpy and extract mel, VERY SLOW
    # if mel, directly load the specs from npz, normalize and return
    if type == "mel":
        train_dataset = BirdClefMelDataset(train=True, args=args, df=train_df, use_secondary=sec, noise_p=0.5)
        valid_dataset = BirdClefMelDataset(train=False, args=args, df=valid_df, use_secondary=sec)
    else:
        raise ("Wrong Dataset type chosen.")

    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_batch_size'],
                              num_workers=4,
                              # prefetch_factor=4,
                              pin_memory=True,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['valid_batch_size'],
                              num_workers=4,
                              # prefetch_factor=4,
                              pin_memory=True,
                              shuffle=True)

    return train_loader, valid_loader


class BirdClefMelDataset(Dataset):
    def __init__(self,
                 train: bool,
                 args,
                 df,
                 aug=None,
                 noise_p=0.2,
                 use_secondary=True
                 ):
        self.train = train
        self.df = df
        self.aug = aug
        self.noise_p = noise_p
        self.secondary = use_secondary
        self.df_paths = df['path']
        self.pri_enc = df['pri_enc']
        self.pri_dec = df['primary_label']
        self.sec_enc = str_array_to_array(df.loc[:, 'sec_enc'])
        self.sr = signal_conf['sr']
        self.hl = signal_conf['hop_length']
        self.dur = signal_conf['len_segment']
        self.dur_samps = int(self.dur * self.sr / self.hl + 1)
        self.noise_files = [f'{args.data_path}/noise/{f}' for f in os.listdir(f'{args.data_path}/noise')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # extract the item chosen
        fpath = pathlib.Path(pathlib.PurePosixPath(self.df_paths[index]))
        specs = np.load(fpath, allow_pickle=True)
        mel = specs.f.mel

        # take a random 30-second chunk
        # pad with zeros if the audio is less than the length
        if mel.shape[1] - self.dur_samps <= 0:
            mel = normalize_0_1(mel)
            pad = np.zeros((mel.shape[0], self.dur_samps - mel.shape[1]))
            mel = np.column_stack((mel, pad))
        # crop a random clip of chosen segment length
        else:
            start = np.random.randint(0, mel.shape[1] - self.dur_samps)
            mel = mel[:, start:start+self.dur_samps]
            mel = normalize_0_1(mel)

        # convert to torch tensor
        mel_normal = torch.FloatTensor(mel)

        # do noise injection with probablitiy noise_p
        if self.noise_p > 0 and np.random.random() < self.noise_p:
            noise_path = random.choice(self.noise_files)
            spec_noise = np.load(noise_path, allow_pickle=True)
            spec_noise = spec_noise.f.mel

            mel_noise = normalize_0_1(torch.FloatTensor(spec_noise))
            mel_noise *= torch.distributions.uniform.Uniform(0.01, 0.5).sample([1])
            mel_normal += mel_noise
            mel_normal = normalize_0_1(mel_normal)

        # add dimensionality layer
        image = mel_normal.unsqueeze(0)
        # load the labels
        if self.secondary:
            label = torch.tensor(self.sec_enc[index]).type(torch.FloatTensor)
        else:
            label = torch.tensor(self.pri_enc[index]).type(torch.LongTensor)

        return image, label


def normalize_0_1(tensor):
    tensor = tensor - tensor.min()
    if tensor.max() != 0:
        tensor /= tensor.max()
    return tensor


def str_array_to_array(str_arr):
    array = []
    for row in str_arr:
        r = row.replace(']', '').replace('[', '')
        r = [int(x) for x in r.split()]
        array.append(r)
    return np.array(array)
