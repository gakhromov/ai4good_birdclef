config = {
    # TRAIN SETTINGS
    "epochs": 25,
    "n_classes": 152,
    "n_folds": 3,
    "num_classes": 152,
    "train_batch_size": 32,
    "valid_batch_size": 32,
    "learning_rate": 1e-4,
    "device": 'cuda',
    "scheduler": 'OneCycle',
    "weight_decay": 1e-6,

    # DATA SETTINGS
    "data_path": 'datasets/birdclef22',
    "mel_path": 'data_processing/numpy_mel',
    "use_secondary": False,
    "sample_rate": 22_050,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "duration": 30,
}

cnn_conf = {
    'filters': [16, 32, 64, 128, 256],
    'kernels': [(3, 3), (3, 5), (3, 5), (3, 5), (3, 5)],
    'strides': [(2, 2), (2, 2), (1, 3), (1, 3), (1, 3)],
    'dense': [256, 256, 256],
    'dropout': True,
    'batch_norm': True,
    'activation': 'relu',
    'regularizer': 0.001
}

signal_conf = {
    "sr": 22050,
    "fmin": 200,
    "fmax": 10000,
    "nmels": 64,
    "nfft": 1024,
    "sr": 22_050,
    "len_segment": 30,
    "lseg_samples": 30 * 22_050
}

wandb_key="13be45bcff4cb1b250c86080f4b3e7ca5cfd29c2"