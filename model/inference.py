from scipy.signal import butter, lfilter
from config import signal_conf
import noisereduce as nr
import librosa
import numpy as np

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
        y = nr.reduce_noise(y=y, sr=sr, stationary=True)

    # extract features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=fmin, fmax=fmax, n_mels=nmels, n_fft=nfft)
    # convert to db and normalize to max=0dB
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec
