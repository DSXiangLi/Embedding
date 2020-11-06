from collections import namedtuple
from utils import get_available_gpus

ALL_DEVICES = get_available_gpus()
NUM_DEVICES = 2

if len(ALL_DEVICES) > NUM_DEVICES:
    ALL_DEVICES = ALL_DEVICES[:NUM_DEVICES]

CHECKPOINT_DIR = './checkpoint/{}_{}'
DICTIONARY_DIR = './data/{}/dictionary.pkl'

TRAIN_PARAMS = {
    'batch_size': 1000 * NUM_DEVICES,
    'epochs': 1000,
    'emb_size': 50,
    'learning_rate': 0.01,
    'ng_sample': 25,
    'buffer_size':128,
    'min_count': 2,
    'decay_steps':-1
}


RUN_CONFIG = {
    'devices': ALL_DEVICES,
    'summary_steps': 50,
    'log_steps': 50,
    'keep_checkpoint_max':3,
    'save_steps': 50
}

SpecialSeqToken = namedtuple('SpecialToken', ['SEQ_START', 'SEQ_END', 'UNK', 'PAD'], defaults= None)
SpecialWordToken = namedtuple('SpecialToken', ['UNK', 'PAD'], defaults= None)

MyWordSpecialToken = SpecialWordToken(UNK = '<UNK>', PAD ='<PAD>')






