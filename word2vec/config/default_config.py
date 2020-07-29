INVALID_INDEX=-1

CHECKPOINT_DIR = './checkpoint/{}_{}'

TRAIN_PARAMS = {
    'batch_size': 100,
    'epochs': 2000,
    'emb_size': 256,
    'learning_rate': 0.01,
    'ng_sample': 25,
    'window_size': 3,
    'buffer_size':128
}