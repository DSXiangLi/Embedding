from config.default_config import TRAIN_PARAMS, MyWordSpecialToken, RUN_CONFIG

INVALID_INDEX = -1

TRAIN_PARAMS_UPDATE = {
    'window_size': 5,
    'sample_rate': 0.001,
    'learning_rate':0.025,
    'ng_sample':5
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)

MySpecialToken = MyWordSpecialToken