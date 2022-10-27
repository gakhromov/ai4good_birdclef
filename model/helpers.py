import torch
from torch.optim import lr_scheduler
import torch
from torch import nn
import numpy as np
import colorednoise as cn
import random

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def fetch_scheduler(optimizer, config):
    """
    It returns a scheduler object based on the string passed in the config file
    
    :param optimizer: the optimizer to use
    :param config: the config file
    :return: The scheduler is being returned.
    """
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    elif config.scheduler == None:
        return None
        
    return scheduler

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def add_white_noise(y, min_snr=5, max_snr=20):
    """
    It adds white noise to the input signal, with a random SNR between 5 and 20 dB
    
    :param y: the signal to be augmented
    :param min_snr: The minimum SNR (signal-to-noise ratio) to add to the signal, defaults to 5
    (optional)
    :param max_snr: The maximum signal-to-noise ratio (SNR) to add to the signal, defaults to 20
    (optional)
    :return: The augmented signal
    """
    snr = np.random.uniform(min_snr, max_snr)
    a_noise = 1 / (10 ** (snr / 20))

    white_noise = np.random.randn(len(y))
    a_white = np.sqrt(white_noise ** 2).max()
    augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
    return augmented


def add_pink_noise(y, min_snr=5, max_snr=20):
    """
    It adds pink noise to the signal, with a random SNR between 5 and 20 dB
    
    :param y: the audio signal
    :param min_snr: minimum signal-to-noise ratio (SNR), defaults to 5 (optional)
    :param max_snr: The maximum signal-to-noise ratio (SNR) to add to the signal, defaults to 20
    (optional)
    :return: The augmented signal.
    """
    snr = np.random.uniform(min_snr, max_snr)
    a_noise = 1 / (10 ** (snr / 20))

    pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
    a_pink = np.sqrt(pink_noise ** 2).max()
    augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
    return augmented  


def add_random_noise(y, max_noise_level=0.5):
    """
    It adds random noise to the input signal
    
    :param y: the original signal
    :param max_noise_level: The maximum amount of noise to add
    :return: The augmented data is being returned.
    """
    noise_level = np.random.uniform(0, max_noise_level)
    noise = np.random.randn(len(y))
    augmented = (y + noise * noise_level).astype(y.dtype)
    return augmented


def add_noise(y, noise_scale=0.5):
    noise_fn = [add_white_noise, add_pink_noise, add_random_noise]
    return random.choice(noise_fn)(y) * noise_scale





