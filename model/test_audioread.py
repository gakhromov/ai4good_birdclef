import torchaudio
import os
import time
import numpy as np
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, PitchShift
import torch
import noisereduce as nr 
import pyrubberband as pyrb
from multiprocessing import Pool
from tqdm import tqdm
from librosa.effects import time_stretch
import argparse

def delete_augment(folder, root):
    files = [f for f in os.listdir(os.path.join(root, folder)) if '_' in f]
    for f in files:
        os.remove(os.path.join(root, folder, f))

def augment_folder(folder, root):
    
    sr=32_000
    shifts = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    rates = [1.05, 0.95, 1.1, 0.9, 1.15, 0.85, 1.2, 0.8, 1.25, 0.75]
    # list files in folders
    files = [f for f in os.listdir(f"{root}/{folder}") if not '_' in f]
    augment = False

    # how many augmentations to make for low sampled birds, maximum augmentations per file are 5
    if len(files) < 20:
        augment = True
        aug_per_file = min(20 // len(files) + 1, 10)

    # load audio and process, then save
    for f in tqdm(files, desc=folder):
        fpath = f"{root}/{folder}/{f}"
        

        # scans audio at original sample rate, return channel / samples
        audio = torchaudio.backend.soundfile_backend.load(fpath)[0]
        # make mono
        audio = torch.mean(audio, dim=0)

        f = fpath.split('.ogg')[0]

        # noisereduce and save original
        ynr = torch.tensor(nr.reduce_noise(y=audio, sr=32_000, stationary=True,  prop_decrease=0.9, n_fft=512)).unsqueeze(0)
        torchaudio.backend.soundfile_backend.save(filepath=f+'_nr.ogg', src=ynr, format='ogg', sample_rate=32_000)

        # make additional augmentations by pitch shift if there are not enough samples
        if augment:
            # make pitch shifts
            for i in range(aug_per_file):
                pitch_shift = PitchShift(sr, shifts[i])
                audio_augmented = pitch_shift(audio)
                ynr = torch.tensor(nr.reduce_noise(y=audio_augmented, sr=32_000, stationary=True,  prop_decrease=0.9, n_fft=512)).unsqueeze(0)
                torchaudio.backend.soundfile_backend.save(filepath=f'{f}_ps{shifts[i]}.ogg', src=ynr, format='ogg', sample_rate=32_000)
            # do time stretch
            for i in range(aug_per_file):
                audio_augmented = time_stretch(audio.numpy(), rate=rates[i])
                ynr = torch.tensor(nr.reduce_noise(y=audio_augmented, sr=32_000, stationary=True,  prop_decrease=0.9, n_fft=512)).unsqueeze(0)
                torchaudio.backend.soundfile_backend.save(filepath=f'{f}_ts{rates[i]}.ogg', src=ynr, format='ogg', sample_rate=32_000)


def main():
    root = '../datasets/birdclef22_augmented/train_audio/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    args = parser.parse_args()
    
    delete_augment(args.folder, root)
    augment_folder(args.folder, root)

    return
    files = os.listdir(root)
    start = time.time()
    melspec = MelSpectrogram(
        sample_rate=32_000,
        n_fft = 1024,
        f_min = 200,
        f_max= 12_000,
        n_mels = 128
    )
    ampdb = AmplitudeToDB(stype="amplitude", top_db=80)
    for file in files:
        f = os.path.join(root, file)
        #print(torchaudio.backend.sox_io_backend.info(f))
        ysox = torchaudio.backend.soundfile_backend.load(f, num_frames=32_000*30)[0]
        # to mono
        ysox = torch.mean(ysox, dim=0)
        ynr = torch.tensor(nr.reduce_noise(y=ysox, sr=32_000, stationary=True,  prop_decrease=0.9, n_fft=512, n_jobs=-1)).unsqueeze(0)
        torchaudio.backend.soundfile_backend.save(filepath=f.split('.ogg')[0]+'_denoise.ogg', src=ynr, format='ogg', sample_rate=32_000)
        # the length is 30 seconds if that's possible, else it is just the real length and must
        # be padded
        ymel = ampdb(melspec(ysox))
    
    print(time.time() - start)
    
if __name__ == "__main__":
    main()