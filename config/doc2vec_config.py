# -*- coding=utf-8 -*-

PARAMS = {'dm': 1,
          'vector_size': 100,
          'window_size': 3,
          'min_count':2,
          'alpha':0.01,
          'min_alpha':0.001,
          'seed':1234,
          'hierarchy_softmax': 1,
          'negative_sampling':0,
          'workers': 4,
          'epochs':50
}

WV_PARAMS = PARAMS.copy()
DV_PARAMS = PARAMS.copy()

WV_PARAMS.update({
    'tagged':False,
    'cbow_mean':1
})

DV_PARAMS.update({
    'infer_epochs':50,
    'tagged':True,
    'dm_mean':1,
    'dm_concat':0,
    'dm_tag_count':1,
    'dbow_words':1
})

CHECKPOINT_DIR = 'checkpoint/doc2vec/{}_{}'
