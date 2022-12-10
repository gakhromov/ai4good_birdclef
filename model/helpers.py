from torch.optim import lr_scheduler
import torch
from torch import nn
import numpy as np
import colorednoise as cn
import random
from config import config
from torchmetrics import Precision, Recall, F1Score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def fetch_scheduler(optimizer, sched: str, spe: int = None, epochs: int = None):
    """
    It returns a scheduler object based on the string passed in the config file
    
    :param optimizer: the optimizer to use
    :param config: the config file
    :return: The scheduler is being returned.
    """
    if sched == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif sched == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
    elif sched == 'OneCycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], steps_per_epoch=spe, epochs=epochs)
    elif sched is None:
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
               (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        loss = loss.mean()
        return loss


class DotDict(dict):
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


def varying_threshold_metrics(preds, targets, targets_thresh=0.2):
    thresholds = np.linspace(0,0.5,20)
    metrics = []
    targets = torch.tensor(targets >= targets_thresh, dtype=torch.long)
    print(targets)
    for thresh in thresholds:
        p = Precision(num_labels=152, threshold=thresh, average='macro', task='multilabel', multidim_average='global')(preds, targets).item()
        r = Recall(num_labels=152, threshold=thresh, average='macro', task='multilabel', multidim_average='global')(preds, targets).item()
        f1 = F1Score(num_labels=152, threshold=thresh, average='macro', task='multilabel', multidim_average='global')(preds, targets).item()
        metrics.append([p, r, f1])
    results = np.column_stack((thresholds, metrics))
    return results
