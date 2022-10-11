config = {
    # TRAIN SETTINGS
    "epochs": 2,
    "n_classes": 10,
    "n_folds": 4,
    "num_classes": 152,
    "train_batch_size": 32,
    "valid_batch_size": 64,
    "learning_rate": 1e-4,
    "device": 'cuda',

    # DATA SETTINGS
    "data_path": '../datasets/birdclef22',
    "sample_rate": 32_000,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "duration": 7,
}
