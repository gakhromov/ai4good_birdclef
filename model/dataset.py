import pandas as pd
from config import config
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pathlib

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
    data_type=args.dtype
    secondary=args.use_secondary

    print(f"Read config {config}")
    df = pd.read_csv(f'{path}/augmented.csv')

    # make folds, stratify with primary labels anyways, even when we have secondary label usage
    skf = StratifiedKFold(n_splits=config['n_folds'])
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label'])):
        df.loc[val_ind, 'fold'] = k
    
    # generate n_fold datasets
    folded_ds = []
    for fold in range(config['n_folds']):
        train_ds, valid_ds = get_data(df, fold, args, type=data_type, sec=secondary)
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
                              shuffle=True)

    return train_loader, valid_loader



class BirdClefMelDataset(Dataset):
    def __init__(self, args, df, aug=None, use_secondary=False):
        self.df = df
        self.df_np = np.load(f'{args.data_path}/augmented.npy', allow_pickle=True)
        self.augmentation = aug
        self.secondary = use_secondary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # extract the item chosen
        fpath = self.df.iloc[index,:]['path']
        fpath = pathlib.Path(pathlib.PurePosixPath(fpath))
        specs = np.load(fpath, allow_pickle=True)
        mel_normal = specs.f.mel
        # Augmentation
        #mel = self.transformation(signal)

        # normalize layers
        mel_normal = mel_normal - np.min(mel_normal)
        if np.max(mel_normal) != 0:
            mel_normal /= np.max(mel_normal)

        # stack layers
        image = mel_normal
        if self.secondary:
            l = self.df_np[index][4]
            label = torch.tensor(l).type(torch.FloatTensor)
        else:
            label = torch.tensor(self.df['pri_enc'][index]).type(torch.LongTensor)
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return image, label
