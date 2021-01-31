# -*- coding=utf-8 -*-
import tensorflow as tf
from config.default_config import SpecialSeqToken, RUN_CONFIG, TRAIN_PARAMS, ModelGN300

MySpecialToken = SpecialSeqToken(SEQ_START='<GO>',
                                 SEQ_END='<EOS>',
                                 PAD='<PAD>',
                                 UNK='<UNK>')

