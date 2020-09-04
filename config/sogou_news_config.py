from config.default_config import TRAIN_PARAMS

INVALID_INDEX = -1

TRAIN_PARAMS_UPDATE = {
    'window_size': 5,
    'sample_rate': 0.01,
    'invalid_index': INVALID_INDEX
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)