import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.signal import butter, lfilter
import pyrubberband as pyrb
import librosa
import sys
import noisereduce as nr
sys.path.append(os.path.abspath('../model'))
from config import signal_conf
from helpers import add_noise

shifts = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
rates = [1.05, 0.95, 1.1, 0.9, 1.15, 0.85, 1.2, 0.8, 1.25, 0.75]
out_folder = 'mel_dataset'
sr = signal_conf['sr']
fmin = signal_conf['fmin']
fmax = signal_conf['fmax']
nmels = signal_conf['nmels']
nfft = signal_conf['nfft']


root = "../datasets/birdclef-2022/train_audio"


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_mel(y, filter=True, reduce_noise=True, return_sig=False):
    # filter with a butterworth bandpass
    if filter:
        y = butter_bandpass_filter(y, fmin, fmax, sr, 3)

    if reduce_noise:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, n_std_thresh_stationary=1.5)
    
    if return_sig:
        return y
    
    # extract features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=fmin, fmax=fmax, n_mels=nmels, n_fft=nfft)
    # convert to db and normalize to max=0dB
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


def convert_audiofolder_to_numpy(folder):
    global sr
    # list files in folders
    files = os.listdir(f"{root}/{folder}")
    augment = False
    os.makedirs(os.path.join(out_folder,'data', folder), exist_ok=True)

    # how many augmentations to make for low sampled birds, maximum augmentations per file are 5
    if len(files) < 20:
        augment = True
        aug_per_file = min(20 // len(files) + 1, 10)

    # load audio and process, then save
    for f in files:
        fpath = f"{root}/{folder}/{f}"
        f = f.split('.')[0]

        # directly get mel , if the file is longer than 30 seconds, split it up in parts
        audio, sr = librosa.load(fpath, sr=sr, res_type='kaiser_fast')
        y = get_mel(audio)
        np.savez_compressed(f"{out_folder}/data/{folder}/{f}", mel=y)

        # make additional augmentations by pitch shift if there are not enough samples
        if augment:
            # make pitch shifts
            for i in range(aug_per_file):
                audio_augmented = pyrb.pitch_shift(audio, sr, shifts[i])
                y = get_mel(audio_augmented)
                np.savez_compressed(f"{out_folder}/data/{folder}/{f}_ps_{shifts[i]}", mel=y)
            # do time stretch
            for i in range(aug_per_file):
                audio_augmented = pyrb.time_stretch(audio, sr, rates[i])
                y = get_mel(audio_augmented)
                np.savez_compressed(f"{out_folder}/data/{folder}/{f}_ts_{rates[i]}", mel=y)


def generate_noise_mels(n_samples):
    root = f"{out_folder}/noise"
    os.makedirs(root, exist_ok=True)

    for i in tqdm(range(n_samples)):
        zeros = np.zeros(sr * 30)
        y = add_noise(zeros)
        mel = get_mel(y, filter=False, reduce_noise=False)
        np.savez_compressed(f'{root}/random_{i}', mel=mel)


def convert_noise_folder(root):
    global sr
    files = [f for f in os.listdir(root) if f.endswith('ogg')]
    for f in tqdm(files):
        fpath = os.path.join(root, f)
        audio, sr = librosa.load(fpath, sr=sr, res_type='kaiser_fast')
        if len(audio) > sr * 30:
            audio = audio[:sr * 30]
        if len(audio) < sr * 30:
            audio = np.hstack((audio, np.zeros(30 * sr - len(audio))))
        y = get_mel(audio, False, False)
        np.savez_compressed(os.path.join(f'{out_folder}/noise', f.split('.')[0]), mel=y)


def main():
    # generating a noise folder, it must always exist
    os.makedirs(f"{out_folder}/noise", exist_ok=True)
    # make the output folder
    os.makedirs(out_folder, exist_ok=True)

    # convert noise to mels
    #print("Converting the provided noise folder...")
    #onvert_noise_folder('../data_processing/aicrowd2020_noise_30sec/noise_30sec')
    print("Generating different noise spectrograms...")
    generate_noise_mels(10) # add as many noise samples as needed, 10 is just for demo


    # scan the train_audio folder and convert all audio files
    folders = [f for f in os.listdir(root)]
    with Pool(4) as p:
        p.map(convert_audiofolder_to_numpy, folders)


if __name__ == "__main__":
    main()
