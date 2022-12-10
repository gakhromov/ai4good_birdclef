import os
import pandas as pd
from config import config, signal_conf
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
import pathlib
from helpers import *
import torch
from albumentations.augmentations import CoarseDropout
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking
from torchaudio.backend import soundfile_backend

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
    skf = StratifiedGroupKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
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
    if type == "mel" and args.mixup:
        train_dataset = BirdClefMixUpMelDataset(train=True, args=args, df=train_df, use_secondary=sec)
        valid_dataset = BirdClefMixUpMelDataset(train=False, args=args, df=valid_df, use_secondary=sec)
    elif type == "mel" and not args.mixup:
        train_dataset = BirdClefMelDataset(train=True, args=args, df=train_df, use_secondary=sec)
        valid_dataset = BirdClefMelDataset(train=False, args=args, df=valid_df, use_secondary=sec)
    else:
        raise ("Wrong Dataset type chosen.")
    # auto fix batch size of slicing
    bs = 1 if config['use_slices'] else config['train_batch_size']
    # generate train loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=bs,
                              num_workers=4,
                              prefetch_factor=4,
                              pin_memory=True,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=bs,
                              num_workers=4,
                              prefetch_factor=4,
                              pin_memory=True,
                              shuffle=False)

    return train_loader, valid_loader


class BirdClefMelDataset(Dataset):
    def __init__(self,
                 train: bool,
                 args,
                 df,
                 noise_p=0.2,
                 use_secondary=True
                 ):
        self.train = train
        self.df = df
        self.aug = CoarseDropout(max_holes=5,max_height=8, max_width=30, p=0.5)
        self.noise_p = noise_p
        self.secondary = use_secondary
        self.df_paths = df['path']
        self.pri_enc = df['pri_enc']
        self.pri_dec = df['primary_label']
        self.scores = df['score']
        self.sec_enc = str_array_to_array(df.loc[:, 'sec_enc'])
        self.sr = signal_conf['sr']
        self.hl = signal_conf['hop_length']
        self.dur = signal_conf['len_segment']
        self.dur_samps = int(self.dur * self.sr / self.hl + 1)
        self.dur_window = (3*self.dur_samps) // 4 # 25% overlap
        self.noise_files = [f'{args.data_path}/noise/{f}' for f in os.listdir(f'{args.data_path}/noise')]
        self.slicing = config['use_slices']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # extract the item chosen
        fpath = pathlib.Path(pathlib.PurePosixPath(self.df_paths[index])).resolve()
        specs = np.load(fpath, allow_pickle=True)
        mel = specs.f.mel
        mel = normalize_0_1(mel)

        # pad with zeros if the audio is not of the length 'dur_samps + k * dur_window, k>=0'
        if mel.shape[1] < self.dur_samps:
            to_pad = self.dur_samps - mel.shape[1] # pad to len = dur_samps
            pad = np.zeros((mel.shape[0], to_pad))
            mel = np.column_stack((mel, pad))
        elif (mel.shape[1] - self.dur_samps) % self.dur_window != 0 and self.slicing:
            residual_time = (mel.shape[1] - self.dur_samps) % self.dur_window
            to_pad = self.dur_window - residual_time # pad to len = dur_samps + k * dur_window
            pad = np.zeros((mel.shape[0], to_pad))
            mel = np.column_stack((mel, pad))

        if self.slicing:
            # take 'dur_samps' chunks with 'dur_window' window
            num_chunks = (mel.shape[1] - self.dur_samps) // self.dur_window + 1
            mels = []
            for chunk in range(num_chunks):
                start = chunk*self.dur_window
                mel_chunk = mel[:, start:start+self.dur_samps]
                mels.append(mel_chunk)
            # convert to torch tensor
            mel_normal = np.array(mels)
        else:
            # pick random 30 second chunk
            if mel.shape[1] == self.dur_samps:
                start = 0
            else:
                start = np.random.randint(0,mel.shape[1] - self.dur_samps)
            mel_normal = mel[:,start:start+self.dur_samps]

        # do noise injection with probablitiy noise_p
        if self.train and np.random.random() < self.noise_p:
            noise_path = random.choice(self.noise_files)
            spec_noise = np.load(noise_path, allow_pickle=True)
            spec_noise = spec_noise.f.mel
            mel_noise = normalize_0_1(spec_noise)
            mel_noise *= np.random.random() * 0.5

            # add noise to the splits
            noise_start = 0
            if not mel_noise.shape[1] == self.dur_samps:
                noise_start = np.random.randint(0, mel_noise.shape[1] - self.dur_samps)
            if self.slicing:
                for split in range(mel_normal.shape[0]):
                    mel_normal[split, :, :] += mel_noise[:, noise_start:noise_start+self.dur_samps]
            else:
                mel_normal += mel_noise[:,noise_start:noise_start+self.dur_samps]

            mel_normal = normalize_0_1(mel_normal)

        # coarse dropout
        mel_normal = self.aug(image=mel_normal)['image']

        # load the labels
        if self.secondary:
            label = torch.tensor(self.sec_enc[index]).type(torch.FloatTensor)
        else:
            label = torch.tensor(self.pri_enc[index]).type(torch.LongTensor)
        
        data = {
            'mels': torch.FloatTensor(mel_normal),
            'path': str(fpath),
            'score': float(self.scores[index])
        }
        return data, label



class BirdClefMixUpMelDataset(Dataset):
    def __init__(self,
                 train: bool,
                 args,
                 df,
                 noise_p=0.5,
                 mixup_p=1.0,
                 use_secondary=False
                 ):
        self.train = train
        self.df = df
        self.noise_p = noise_p
        self.mixup_p = mixup_p
        self.secondary = use_secondary
        self.df_paths = df['path']
        self.pri_enc = df['pri_enc']
        self.pri_dec = df['primary_label']
        self.scores = df['score']
        self.sec_enc = str_array_to_array(df.loc[:, 'sec_enc'])
        self.sr = signal_conf['sr']
        self.hl = signal_conf['hop_length']
        self.dur = signal_conf['len_segment']
        self.dur_samps = int(self.dur * self.sr / self.hl + 1)
        self.noise_files = [f'{args.data_path}/noise/{f}' for f in os.listdir(f'{args.data_path}/noise')]
        self.freq_masking = FrequencyMasking(freq_mask_param=12)
        self.time_masking = TimeMasking(time_mask_param=128)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # load the mel spectogram
        fpath = pathlib.Path(pathlib.PurePosixPath(self.df_paths[index])).resolve()
        specs = np.load(fpath, allow_pickle=True)
        mel = specs.f.mel
        mel = pad_mel(normalize_0_1(mel), self.dur_samps)
        mel = sample_chunk(mel, self.dur_samps)
        # load label
        label = torch.tensor(self.sec_enc[index]).type(torch.FloatTensor)

        # do mixup 
        if self.train and np.random.random() < self.mixup_p:
            index_mix = np.random.randint(0, len(self.df))
            fpath = pathlib.Path(pathlib.PurePosixPath(self.df_paths[index_mix])).resolve()
            specs = np.load(fpath, allow_pickle=True)
            mel_mix = specs.f.mel
            mel_mix = pad_mel(normalize_0_1(mel_mix), self.dur_samps)
            mel_mix = sample_chunk(mel_mix, self.dur_samps)
            # load label
            label_mix = torch.tensor(self.sec_enc[index_mix]).type(torch.FloatTensor)

            # do mixup
            r = np.clip(np.random.random(), 0.3, 0.7)
            mel = mel * r + mel_mix * (1 - r) #TODO make better scaling according to amplitude
            label = label*r + label_mix*(1-r)
            # clamp the label, add positive label smoothing
            label = label.clamp(0,1)
        

        # add noise to it
        if self.train and np.random.random() < self.noise_p:
            noise_path = random.choice(self.noise_files)
            spec_noise = np.load(noise_path, allow_pickle=True)
            spec_noise = spec_noise.f.mel
            mel_noise = np.random.random() * 0.5 * normalize_0_1(spec_noise) #TODO maybe sample differently
            mel_noise = sample_chunk(mel_noise, self.dur_samps)
            mel += mel_noise
            mel = normalize_0_1(mel)


        if self.train:
            # do augmentations
            mel = torch.from_numpy(mel).unsqueeze(0)
            mel = self.freq_masking(mel) # masking needs a batch dimension at dim 0
            mel = self.time_masking(mel).squeeze()
            
        data = {
            'mels': mel,
            'path': str(self.df_paths[index]),
            'score': float(self.scores[index])
        }

        return data, label
    

    def preprocess_sample(self, index):
        s = self.df.iloc[index, :]
        path1 = s['path']
        length = s['length']
        label = self.sec_enc[index]
        label = torch.tensor(label).type(torch.FloatTensor)
        start = 0 if length <= self.dur * self.sr else np.random.randint(0, length - self.dur*self.sr)
        # read audio and make mono
        y = soundfile_backend.load(path1, frame_offset=start, num_frames=32_000*30)[0]
        y = torch.mean(y, dim=0) 
        # padding
        if len(y) < self.dur*self.sr:
            y = torch.cat((y, torch.zeros(self.dur*self.sr - len(y))), dim=-1)
        y = normalize_plus_minus_1(y)
        return y, label


class BirdClefMixUpOggDataset(Dataset):
    def __init__(self,
                 train: bool,
                 args,
                 df,
                 noise_p=0.2,
                 use_secondary=False
                 ):
        self.train = train
        self.df = df
        self.noise_p = noise_p
        self.secondary = use_secondary
        self.df_paths = df['path']
        self.pri_enc = df['pri_enc']
        self.pri_dec = df['primary_label']
        self.scores = df['score']
        self.sec_enc = str_array_to_array(df.loc[:, 'sec_enc'])
        self.sr = signal_conf['sr']
        self.hl = signal_conf['hop_length']
        self.dur = signal_conf['len_segment']
        self.noise_files = [f'{args.data_path}/noise/{f}' for f in os.listdir(f'{args.data_path}/noise')]
        self.ampdb = AmplitudeToDB(stype="power", top_db=80)
        self.melspec = MelSpectrogram(
            sample_rate=32_000,
            n_fft = 1024,
            f_min = 200,
            f_max= 12_000,
            n_mels = 128,
            hop_length=512,
            normalized=True
            )
        self.freq_masking = FrequencyMasking(freq_mask_param=12)
        self.time_masking = TimeMasking(time_mask_param=128)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # do mixup with audio
        if self.train:
            y1, label1 = self.preprocess_sample(index)
            index_2 = np.random.randint(0, len(self.df))
            y2, label2 = self.preprocess_sample(index_2)
            # do mixup
            r = np.random.random()
            sound = y1 * r + y2 * (1 - r) #TODO make better scaling according to amplitude
            label = label1*r + label2*(1-r)
            # clamp the label, add positive label smoothing
            label = label.clamp(0.01,1)
        else:
            sound, label = self.preprocess_sample(index)
        
        # calculate mel spectrum
        mel = self.melspec(sound)

        # AUGMENTATIONS
        if self.train:
            mel = self.freq_masking(mel.unsqueeze(0))
            mel = self.time_masking(mel)
            mel = mel.squeeze()

        # convert to db
        mel = self.ampdb(mel)
        # normalize to 0-1 range
        mel = normalize_0_1(mel)

        data = {
            'mels': mel,
            'path': str(self.df_paths[index]),
            'score': float(self.scores[index])
        }

        return data, label
    

    def preprocess_sample(self, index):
        s = self.df.iloc[index, :]
        path1 = s['path']
        length = s['length']
        label = self.sec_enc[index]
        label = torch.tensor(label).type(torch.FloatTensor)
        start = 0 if length <= self.dur * self.sr else np.random.randint(0, length - self.dur*self.sr)
        # read audio and make mono
        y = soundfile_backend.load(path1, frame_offset=start, num_frames=32_000*30)[0]
        y = torch.mean(y, dim=0) 
        # padding
        if len(y) < self.dur*self.sr:
            y = torch.cat((y, torch.zeros(self.dur*self.sr - len(y))), dim=-1)
        y = normalize_plus_minus_1(y)
        return y, label


def sample_chunk(mel, duration):
    assert mel.shape[1] >= duration, "Mel was not padded enough!"
    # pick random 30 second chunk
    if mel.shape[1] == duration:
        return mel
    else:
        start = np.random.randint(0,mel.shape[1] - duration)
        mel = mel[:,start:start+duration]
    return mel
    
def pad_mel(mel, duration):
    if mel.shape[1] < duration:
        to_pad = duration - mel.shape[1] # pad to len = dur_samps
        pad = np.zeros((mel.shape[0], to_pad))
        mel = np.column_stack((mel, pad))
    return mel

def normalize_0_1(tensor):
    tensor = tensor - tensor.min()
    if tensor.max() != 0:
        tensor /= tensor.max()
    return tensor


def normalize_plus_minus_1(tensor):
    max_val = np.max((-tensor.min(), tensor.max()))
    if max_val != 0:
        tensor /= max_val
    return tensor


def mix(sound1, sound2, r, fs):
    sound = sound1 * r + sound2 * (1 - r)
    return sound


def str_array_to_array(str_arr):
    array = []
    for row in str_arr:
        r = row.replace(']', '').replace('[', '')
        r = [int(x) for x in r.split()]
        array.append(r)
    return np.array(array)
