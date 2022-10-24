config = {
    # TRAIN SETTINGS
    "epochs": 2,
    "n_classes": 152,
    "n_folds": 4,
    "num_classes": 152,
    "train_batch_size": 2,
    "valid_batch_size": 2,
    "learning_rate": 1e-4,
    "device": 'cuda',

    # DATA SETTINGS
    "data_path": 'datasets/birdclef22',
    "mel_path": 'data_processing/numpy_mel',
    "sample_rate": 32_000,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "duration": 30,
}

cnn_conf = {
    'filters': [16, 32, 64, 128, 256],
    'kernels': [(3,3), (3,5), (3,5), (3,5), (3,5)],
    'strides': [(2,2), (2,2), (1,3), (1,3), (1,3)],
    'dense': [32, 16, 8],
    'dropout': True,
    'batch_norm': True,
    'activation': 'relu',
    'regularizer': 0.001
}


class CFG:
    scheduler = 'CosineAnnealingWarmRestarts'
    T_max = 10
    T_0 = 10
    lr = 1e-4
    weight_decay = 1e-6
    lr = 1e-4
    min_lr = 1e-6
    num_classes = 152
