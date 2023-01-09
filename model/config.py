config = {
    # TRAIN SETTINGS
    "epochs": 50,
    "n_classes": 152,
    "n_folds": 3,
    "num_classes": 152,
    "train_batch_size": 2,
    "learning_rate": 1e-3,
    "device": 'cuda',
    "scheduler": 'OneCycle',
    "weight_decay": 1e-7,
    "ast": True, # Training with Audio Spectrogram Transformer
    # DATA SETTINGS
    "use_secondary": False, # use secondary labels => multi-label training
    "use_slices": False, # use slicing => trains on whole recordings
    "mixup": False
}

cnn_conf = {
    'filters': [32, 64, 128, 256, 512],
    'kernels': [(3, 3), (3, 5), (3, 5), (3, 5), (3, 5)],
    'strides': [(2, 2), (2, 2), (1, 3), (1, 3), (1, 3)],
    'dense': [512, 256],
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

wandb_key = None
