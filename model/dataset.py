import pandas as pd
import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import torch.nn.functional as F


def transform(tsf="mel_spec"):
    if tsf == "mel_spec":
        return torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                    n_fft=config.n_fft,
                                                    hop_length=config.hop_length,
                                                    n_mels=config.n_mels)


def get_dataset():
    # read the metadata
    df = pd.read_csv(f'{config.data_path}/train_metadata.csv')
    # encode labels
    encoder = LabelEncoder()
    df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])
    # make folds
    skf = StratifiedKFold(n_splits=config.n_folds)
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label_encoded'])):
        df.loc[val_ind, 'fold'] = k

    # generate n_fold datasets
    folded_ds = []
    for fold in range(config.n_folds):
        train_ds, valid_ds = get_data(df, fold)
        folded_ds.append((train_ds, valid_ds))

    return folded_ds


def get_data(df, fold):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    train_dataset = BirdClefDataset(train_df, transform("mel_spec"), config.sample_rate, config.duration)
    valid_dataset = BirdClefDataset(valid_df, transform("mel_spec"), config.sample_rate, config.duration)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False)

    return train_loader, valid_loader


class BirdClefDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration):
        self.audio_paths = df['filename'].values
        self.labels = df['primary_label_encoded'].values
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = f'{config.data_path}/train_audio/{self.audio_paths[index]}'
        signal, sr = torchaudio.load(audio_path)  # loaded the audio

        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if it not equal we perform
        # resampling
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)

        # Next we check the number of channels of the signal
        # signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0, keepdim=True)

        # Lastly we check the number of samples of the signal
        # signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]

        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)

        # Finally all the process has been done and now we will extract mel spectrogram from the signal
        mel = self.transformation(signal)

        # For pretrained models, we need 3 channel image, so for that we concatenate the extracted mel
        # image = torch.cat([mel, mel, mel])
        image = mel

        # Normalized the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.tensor(self.labels[index])

        return image, label
