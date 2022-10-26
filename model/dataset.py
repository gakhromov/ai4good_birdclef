import pandas as pd
from config import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import librosa
import noisereduce as nr
import glob
import random

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
    data_folder=args.data_folder
    data_type=args.dtype
    secondary=args.use_secondary

    # read the metadata
    print(f"Read config {config}")
    df = pd.read_csv(f"{path}/train_metadata.csv")

    # encode primary labels
    encoder = LabelEncoder()
    primary = encoder.fit_transform(df['primary_label'])
    df['pri_enc'] = primary
    # encode secondary labels
    if secondary:
        classes = df['primary_label'].unique()
        df['label'] = [combine_labels(df['primary_label'][idx], df['secondary_labels'][idx]) for idx in range(len(df))]
        secondary = [np.sum([np.where(item == classes, 1, 0) for item in row], axis=0) for row in df['label']]
        df['sec_enc'] = secondary 

    # make folds, stratify with primary labels anyways, even when we have secondary label usage
    skf = StratifiedKFold(n_splits=config['n_folds'])
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=primary)):
        df.loc[val_ind, 'fold'] = k
    
    # generate n_fold datasets
    folded_ds = []
    for fold in range(config['n_folds']):
        train_ds, valid_ds = get_data(df, fold, data_folder, type=data_type, sec=secondary)
        folded_ds.append((train_ds, valid_ds))

    return folded_ds


def get_data(df, fold, data_folder, type="ogg", sec=False):
    # extract fold
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    # if ogg, load the data from ogg files, convert to numpy and extract mel, VERY SLOW
    # if mel, directly load the specs from npz, normalize and return
    if type=="ogg":
        train_dataset = BirdClefOggDataset(df=train_df, path=data_folder, sr=config['sample_rate'], duration=config['duration'])
        valid_dataset = BirdClefOggDataset(df=valid_df, path=data_folder, sr=config['sample_rate'], duration=config['duration'])
    elif type=="mel":
        train_dataset = BirdClefMelDataset(df=train_df, mel_path=data_folder, sr=config['sample_rate'], duration=config['duration'], use_secondary=sec)
        valid_dataset = BirdClefMelDataset(df=valid_df, mel_path=data_folder, sr=config['sample_rate'], duration=config['duration'], use_secondary=sec)
    else:
        exit("Wrong Dataset type chosen.")


    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_batch_size'],
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=4,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['valid_batch_size'],
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=4,
                              shuffle=False)

    return train_loader, valid_loader



class BirdClefMelDataset(Dataset):
    def __init__(self, df, mel_path, sr, duration, aug=None, use_secondary=False):

        # extract the classes and filenames
        folders = os.listdir(mel_path)
        filenames = []
        for f in folders:
            files = [f'{mel_path}/{f}/{file}' for file in os.listdir(os.path.join(mel_path, f))]
            filenames.append(files)

        self.df = df
        self.mel_path = mel_path
        self.augmentation = aug
        self.sr = sr
        self.num_samples = sr * duration // config['n_fft']
        self.secondary = use_secondary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # extract the item chosen
        item = self.df.iloc[index,:]
        root = f"{self.mel_path}/{item['filename'].split('.')[0]}*"
        # random file which matches the root
        fpath = glob.glob(root)

        # load one random choice of the compressed file
        specs = np.load(random.choice(fpath), allow_pickle=True)
        mel_normal = specs.f.mel


        # Augmentation
        #mel = self.transformation(signal)

        # normalize layers
        mel_normal = mel_normal - np.min(mel_normal)
        mel_normal = mel_normal / np.max(mel_normal)

        # stack layers
        image = mel_normal
        if self.secondary:
            label = torch.tensor(self.df['sec_enc'][index]).type(torch.LongTensor)
        else:
            label = torch.tensor(self.df['pri_enc'][index]).type(torch.LongTensor)
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return image, label


class BirdClefOggDataset(Dataset):
    def __init__(self, df, path, sr, duration, aug=None, fmin=100, fmax=12000, nmel=64, fftlength=1024):
        # extract the filenames
        self.paths = [os.path.join(path, fn) for fn in df['filename'].values]
        self.labels = df['primary_label_encoded'].values
        self.augmentation = aug
        self.sr = sr
        self.num_samples = sr * duration // config['n_fft']
        self.fmin = fmin
        self.fmax=fmax
        self.nm = nmel
        self.fftl = fftlength

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # load audio, atm only first 30 seconds
        y, sr = librosa.load(path=path, duration=30, sr=self.sr, res_type='kaiser_fast')

        # denoise
        y = nr.reduce_noise(y, sr=self.sr, stationary=True)

        # calculate mel spec
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, fmin=self.fmin, fmax=self.fmax, n_mels=self.nm, n_fft=self.fftl)
        # convert to db
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # zero pad if needed
        # Printing the shape of the mel_normal array.
        difference = self.num_samples - mel_spec.shape[1]

        if difference > 0:
            padding = np.zeros((mel_spec.shape[0], difference))
            mel_spec = np.hstack((mel_spec, padding))
        if difference < 0:
            mel_spec = mel_spec[:,:self.num_samples]


        # Augmentation
        #mel = self.transformation(signal)

        # normalize layers
        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)

        # stack layers
        image = mel_spec

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index]).type(torch.LongTensor)

        return image, label
