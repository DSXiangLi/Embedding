import tensorflow as tf
import os
from collections import namedtuple
from utils import get_available_gpus

ALL_DEVICES = get_available_gpus()
NUM_DEVICES = 3

if len(ALL_DEVICES) > NUM_DEVICES:
    ALL_DEVICES = ALL_DEVICES[-NUM_DEVICES:]

CHECKPOINT_DIR = './checkpoint/{}_{}'
DICTIONARY_DIR = './data/{}/dictionary.pkl'

TRAIN_PARAMS = {
    'batch_size': 128 * max(1, len(ALL_DEVICES)),
    'epochs': 1000,
    'emb_size': 50,
    'learning_rate': 0.01,
    'ng_sample': 25,
    'buffer_size':128,
    'min_count': 5,
    'max_count': 10000,
    'decay_steps': -1,
    'decay_rate': 0.01,
    'lower_gradient': -10,
    'upper_gradient': 10,
}


RUN_CONFIG = {
    'devices': ALL_DEVICES,
    'summary_steps': 50,
    'log_steps': 50,
    'keep_checkpoint_max':3,
    'save_steps': 50,
    'allow_growth': True,
    'pre_process_gpu_fraction': 0.8,
    'log_device_placement': True,
    'allow_soft_placement': True,
    'inter_op_parallel': 2,
    'intra_op_parallel': 2
}


def set_encoder_decoder_params(model, params, ed_params):
    """
    model name consist of 'encode_decoder'
    - cnn/gru are supported for encoder
    - gru/lstm are supported for decoder
    """
    encoder, decoder = model.split('_')

    params['decoder_cell'] = decoder
    params['encoder_cell'] = encoder
    params['encoder_type'] = '{}_encoder'.format(encoder)
    params['decoder_type'] = '{}_decoder'.format(decoder)
    params['encoder_cell_params'] = ed_params[encoder]
    params['decoder_cell_params'] = ed_params[decoder]

    return params


SpecialSeqToken = namedtuple('SpecialToken', ['SEQ_START', 'SEQ_END', 'UNK', 'PAD'])
SpecialWordToken = namedtuple('SpecialToken', ['UNK', 'PAD'])


PretrainModelDir = './data/pretrain_model'
ModelGN300 = os.path.join(PretrainModelDir, 'GoogleNews-vectors-negative300.bin')
