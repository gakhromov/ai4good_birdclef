config = {
    # TRAIN SETTINGS
    "epochs": 50,
    "n_classes": 152,
    "n_folds": 2,
    "num_classes": 152,
    "train_batch_size": 1,
    "valid_batch_size": 1,
    "learning_rate": 1e-4,
    "device": 'cuda',
    "scheduler": 'OneCycle',
    "weight_decay": 1e-6,

    # DATA SETTINGS
    "data_path": 'datasets/birdclef22',
    "mel_path": 'data_processing/numpy_mel',
    "use_secondary": False,
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
    "sr": 32_000,
    "fmin": 200,
    "fmax": 10000,
    "nmels": 128,
    "nfft": 1024,
    "len_segment": 30,
    "hop_length": 512
}

wandb_key="13be45bcff4cb1b250c86080f4b3e7ca5cfd29c2"