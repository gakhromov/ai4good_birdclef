import pandas as pd
from config import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import os
import librosa
import noisereduce as nr

def get_dataset(path: str, data_folder: str):
    # read the metadata
    print(f"Read config {config}")
    df = pd.read_csv(f"{path}/train_metadata.csv")

    # encode labels
    encoder = LabelEncoder()
    df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])

    # make folds
    skf = StratifiedKFold(n_splits=config['n_folds'])
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label_encoded'])):
        df.loc[val_ind, 'fold'] = k

    # generate n_fold datasets
    folded_ds = []
    for fold in range(config['n_folds']):
        train_ds, valid_ds = get_data(df, fold, data_folder)
        folded_ds.append((train_ds, valid_ds))

    return folded_ds


def get_data(df, fold, data_folder, type="ogg"):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    if type=="ogg":
        train_dataset = BirdClefOggDataset(df=train_df, path=data_folder, sr=config['sample_rate'], duration=config['duration'])
        valid_dataset = BirdClefOggDataset(df=valid_df, path=data_folder, sr=config['sample_rate'], duration=config['duration'])
    if type=="mel":
        train_dataset = BirdClefMelDataset(train_df, data_folder, config['sample_rate'], config['duration'])
        valid_dataset = BirdClefMelDataset(valid_df, data_folder, config['sample_rate'], config['duration'])
    else:
        exit("Wrong Dataset type chosen.")


    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_batch_size'],
                              num_workers=4,
                              pin_memory=False,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['valid_batch_size'],
                              num_workers=4,
                              pin_memory=False,
                              shuffle=False)

    return train_loader, valid_loader

class BirdClefMelDataset(Dataset):
    def __init__(self, df, path, sr, duration, aug=None):
        self.mel_paths = [os.path.join(path, fn) for fn in df['filename'].values]
        self.labels = df['primary_label_encoded'].values
        self.augmentation = aug
        self.sr = sr
        self.num_samples = sr * duration // config['n_fft']

    def __len__(self):
        return len(self.mel_paths)

    def __getitem__(self, index):
        # change index from / to _
        path = self.mel_paths[index]
        path = path.replace('ogg', 'npz')

        # load the compressed file
        specs = np.load(path, allow_pickle=True)
        mel_normal = specs.f.original

        # zero pad if needed
        # Printing the shape of the mel_normal array.
        difference = self.num_samples - mel_normal.shape[1]

        if difference > 0:
            padding = np.zeros((mel_normal.shape[0], difference))
            mel_normal = np.hstack((mel_normal, padding))
        if difference < 0:
            mel_normal = mel_normal[:,:self.num_samples]


        # Augmentation
        #mel = self.transformation(signal)

        # normalize layers
        mel_normal = mel_normal - np.min(mel_normal)
        mel_normal = mel_normal / np.max(mel_normal)

        # stack layers
        image = mel_normal

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index]).type(torch.LongTensor)

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
