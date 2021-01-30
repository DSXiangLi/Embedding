# -*- coding=utf-8 -*-

import pickle
import numpy as np

def normalize(vector: np.ndarray):
    norm = np.linalg.norm(vector)
    if norm == 0:
        norm = np.finfo(vector.dtype).eps
    return vector / norm


with open('./data/bookcorpus/skip_thought_predict.pkl', 'rb') as f:
    res = pickle.load(f)


seqs = np.array([' '.join( [i.decode('utf-8') for i in item['input_token']]) for item in res])
sentence_embedding = dict(zip(seqs, [item['vector'] for item in res]))

all_embedding = np.array([item['vector'] for item in res ])
all_embedding = np.apply_along_axis(normalize, 1, all_embedding)


def find_topn_most_similar(sentence, topn):
    emb = sentence_embedding[sentence]

    emb = normalize(emb)

    sim = np.matmul(np.expand_dims(emb, axis=0), all_embedding.transpose())

    items = np.squeeze(np.argsort(sim, axis=1)[:, (-topn-1):-1])

    scores = np.squeeze(sim[:, items])

    sim_seq = seqs[items]

    for i in range(topn):
        print('Most Similar score = {:.2f} sentences = {}'.format(scores[i], sim_seq[i]))

topn = 2

for i in range(50):
    print('\n Finding Top {} for "{}" '.format(topn, seqs[i]))

    find_topn_most_similar(seqs[i], topn)


