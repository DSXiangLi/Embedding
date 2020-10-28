from collections import namedtuple

CHECKPOINT_DIR = './checkpoint/{}_{}'

TRAIN_PARAMS = {
    'batch_size': 200,
    'epochs': 1000,
    'emb_size': 50,
    'learning_rate': 0.01,
    'ng_sample': 25,
    'buffer_size':128,
    'min_count': 2,
    'decay_steps':-1
}


RUN_CONFIG = {
    'summary_steps': 50,
    'log_steps': 50,
    'keep_checkpoint_max':3,
    'save_steps': 50
}

SpecialSeqToken = namedtuple('SpecialToken', ['SEQ_START', 'SEQ_END', 'UNK', 'PAD'], defaults= None)
SpecialWordToken = namedtuple('SpecialToken', ['UNK', 'PAD'], defaults= None)

MyWordSpecialToken = SpecialWordToken(UNK = '<UNK>', PAD ='<PAD>')
