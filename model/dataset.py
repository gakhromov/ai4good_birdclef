import pandas as pd
from config import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import os


def get_dataset():
    # read the metadata
    print(f"Read config {config}")
    df = pd.read_csv(f"{config['data_path']}/train_metadata.csv")
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
        train_ds, valid_ds = get_data(df, fold)
        folded_ds.append((train_ds, valid_ds))

    return folded_ds


def get_data(df, fold):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    train_dataset = BirdClefDataset(train_df, config['sample_rate'], config['duration'])
    valid_dataset = BirdClefDataset(valid_df, config['sample_rate'], config['duration'])

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

class BirdClefDataset(Dataset):
    def __init__(self, df, sr, duration, aug=None):
        self.mel_paths = df['filename'].values
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
        specs = np.load(os.path.join(config['mel_path'], path), allow_pickle=True)
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
