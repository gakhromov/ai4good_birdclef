import pandas as pd
from config import config, signal_conf
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pathlib
from helpers import *
import torch, torchaudio
from models import MelDB


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
    path=args.data_path
    secondary=args.use_secondary

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


def get_data(df, fold, args, type="mel", sec=False):
    # extract fold
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    # if ogg, load the data from ogg files, convert to numpy and extract mel, VERY SLOW
    # if mel, directly load the specs from npz, normalize and return
    if type=="mel":
        train_dataset = BirdClefMelDataset(args, df=train_df, use_secondary=sec)
        valid_dataset = BirdClefMelDataset(args, df=valid_df, use_secondary=sec)
    else:
        raise("Wrong Dataset type chosen.")


    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_batch_size'],
                              num_workers=4,
                              prefetch_factor=4,
                              pin_memory=True,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['valid_batch_size'],
                              num_workers=4,
                              prefetch_factor=4,
                              pin_memory=True,
                              shuffle=True)

    return train_loader, valid_loader


class BirdClefMelDataset(Dataset):
    def __init__(self, args, df, aug=None, noise_p=1, use_secondary=False):
        self.df = df
        self.df_np = np.load(f'{args.data_path}/augmented.npy', allow_pickle=True)
        self.aug = aug
        self.mel = MelDB().to(config['device'])
        self.noise_p = noise_p
        self.secondary = False
        self.df_paths = df['path']
        self.pri_enc = df['pri_enc']
        self.pri_dec = df['primary_label']
        self.sr = signal_conf['sr']
        self.dur = signal_conf['len_segment']
        self.mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=22_050,
                    n_fft=1024,
                    f_min=200,
                    f_max=10_000,
                    hop_length=512,
                    n_mels=64,
                    normalized=True)
        self.atodb = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # extract the item chosen
        fpath = pathlib.Path(pathlib.PurePosixPath(self.df_paths[index]))
        specs = np.load(fpath, allow_pickle=True)

        mel_normal = torch.FloatTensor(specs.f.mel)
        # do noise injection with probablitiy noise_p
        if self.noise_p > 0:
            if np.random.random() < self.noise_p:
                noise = torch.FloatTensor(add_noise(np.zeros(self.sr*self.dur), noise_scale=0.5))
                mel_normal += self.atodb(self.mel(noise))

        # normalize layers
        mel_normal = mel_normal - mel_normal.min()
        if mel_normal.max() != 0:
            mel_normal /= mel_normal.max()

        # stack layers (will come later)
        image = mel_normal.unsqueeze(0)

        if self.secondary:
            l = self.df_np[index][4]
            label = torch.tensor(l).type(torch.FloatTensor)
        else:
            label = torch.tensor(self.pri_enc[index]).type(torch.LongTensor)

        return image, label
