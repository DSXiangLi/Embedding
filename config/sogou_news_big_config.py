from config.default_config import TRAIN_PARAMS, SpecialWordToken, RUN_CONFIG

TRAIN_PARAMS_UPDATE = {
    'window_size': 5,
    'sample_rate': 0.01,
    'batch_size': 500,
    'epochs': 1000,
    'emb_size': 200
}

TRAIN_PARAMS.update(TRAIN_PARAMS_UPDATE)

MySpecialToken = SpecialWordToken(UNK = '<UNK>', PAD ='<PAD>')