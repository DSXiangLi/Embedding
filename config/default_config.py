INVALID_INDEX = -1

CHECKPOINT_DIR = './checkpoint/{}_{}'

TRAIN_PARAMS = {
    'batch_size': 200,
    'epochs': 1000,
    'emb_size': 50,
    'learning_rate': 0.01,
    'ng_sample': 25,
    'buffer_size':128,
    'min_count': 2,
    'decay_steps':-1,
    'invalid_index': INVALID_INDEX
}


RUN_CONFIG = {
    'summary_steps': 50,
    'log_steps': 50,
    'keep_checkpoint_max':3,
    'save_steps': 50
}