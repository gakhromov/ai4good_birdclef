import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.signal import butter, lfilter
import pyrubberband as pyrb
import librosa
import noisereduce as nr
import sys
print(os.path.relpath('..'))
sys.path.append(os.path.relpath('..'))
from model.config import signal_conf
from model.helpers import add_noise


shifts = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
rates = [1.05, 0.95, 1.1, 0.9, 1.15, 0.85, 1.2, 0.8, 1.25, 0.75]
out_folder = 'numpy_mel_segments'
sr = signal_conf['sr']
fmin = signal_conf['fmin']
fmax = signal_conf['fmax']
nmels = signal_conf['nmels']
nfft = signal_conf['nfft']


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_mel(y, filter=True, reduce_noise=True):
    # filter with a butterworth bandpass
    if filter:
        y = butter_bandpass_filter(y, fmin, fmax, sr, 3)

    if reduce_noise:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, n_std_thresh_stationary=1.5)

    # extract features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=fmin, fmax=fmax, n_mels=nmels, n_fft=nfft)
    # convert to db and normalize to max=0dB
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


def convert_audiofolder_to_numpy(folder):
    global sr
    # list files in folders
    files = os.listdir(f"../datasets/birdclef22/train_audio/{folder}")
    augment = False
    os.makedirs(os.path.join(out_folder, folder), exist_ok=True)

    # how many augmentations to make for low sampled birds, maximum augmentations per file are 5
    if len(files) < 20:
        augment = True
        aug_per_file = min(20 // len(files) + 1, 10)

    # load audio and process, then save
    for f in files:
        fpath = f"../datasets/birdclef22/train_audio/{folder}/{f}"
        f = f.split('.')[0]

        # directly get mel , if the file is longer than 30 seconds, split it up in parts
        audio, sr = librosa.load(fpath, sr=sr, res_type='kaiser_fast')
        y = get_mel(audio)
        np.savez_compressed(f"{out_folder}/{folder}/{f}", mel=y)

        # make additional augmentations by pitch shift if there are not enough samples
        if augment:
            # make pitch shifts
            for i in range(aug_per_file):
                audio_augmented = pyrb.pitch_shift(audio, sr, shifts[i])
                y = get_mel(audio_augmented)
                np.savez_compressed(f"{out_folder}/{folder}/{f}_ps_{shifts[i]}", mel=y)
            # do time stretch
            for i in range(aug_per_file):
                audio_augmented = pyrb.time_stretch(audio, sr, rates[i])
                y = get_mel(audio_augmented)
                np.savez_compressed(f"{out_folder}/{folder}/{f}_ts_{rates[i]}", mel=y)


def npz_to_npy(folders):
    root = '../datasets/numpy_mel/data'
    for folder in folders:
        files = os.listdir(f"{root}/{folder}")
        for file in files:
            fpath = os.path.join(root, folder, file)
            dat = np.load(fpath, allow_pickle=True)
            dat = dat.f.mel
            np.save(f"{root}/{folder}/{file.replace('npz', 'npy')}", dat)


def generate_noise_mels(n_samples):
    root = '../datasets/numpy_mel/noise'
    os.makedirs(root, exist_ok=True)

    for i in tqdm(range(n_samples)):
        zeros = np.zeros(22_050 * 30)
        y = add_noise(zeros)
        mel = get_mel(y, filter=False, reduce_noise=False)
        np.savez_compressed(f'{root}/random_{i}', mel=mel)


def convert_noise_folder(root='../data_processing/aicrowd2020_noise_30sec/noise_30sec'):
    files = [f for f in os.listdir(root) if f.endswith('ogg')]
    sr = 22_050
    for f in tqdm(files):
        fpath = os.path.join(root, f)
        audio, sr = librosa.load(fpath, sr=sr, res_type='kaiser_fast')
        if len(audio) > sr * 30:
            audio = audio[:sr * 30]
        if len(audio) < sr * 30:
            audio = np.hstack((audio, np.zeros(30 * sr - len(audio))))
        y = get_mel(audio, False, False)
        np.savez_compressed(os.path.join(f'../datasets/numpy_mel/noise', f.split('.')[0]), mel=y)


if __name__ == "__main__":
    # convert_noise_folder()
    # generate_noise_mels(1000)

    # make the output folder
    os.makedirs(out_folder, exist_ok=True)
    # scan the train_audio folder
    folders = [f for f in os.listdir("../datasets/birdclef22/train_audio")]
    # convert_audiofolder_to_numpy(folders[0])

    with Pool(4) as p:
        p.map(convert_audiofolder_to_numpy, folders)
