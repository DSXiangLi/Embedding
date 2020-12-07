# -*- coding=utf-8 -*-

import pickle
import numpy as np

def normalize(vector: np.ndarray):
    norm = np.linalg.norm(vector)
    if norm == 0:
        norm = np.finfo(vector.dtype).eps
    return vector / norm


with open('./data/bookcorpus/quick_thought_predict_embedding.pkl', 'rb') as f:
    res = pickle.load(f)


seqs = np.array(list(res.keys()))
embedding = np.array(list(res.values()))
embedding = np.apply_along_axis(normalize, 1, embedding)


def find_topn_most_similar(sentence, topn):
    emb = np.array(res[sentence])

    emb = normalize(emb)

    sim = np.matmul(np.expand_dims(emb, axis=0), embedding.transpose())

    items = np.squeeze(np.argsort(sim, axis=1)[:, -topn:])

    scores = np.squeeze(sim[:, items])

    sim_seq = seqs[items]

    for i in range(topn):
        print('Most Similar score = {:.2f} sentences = {}'.format(scores[i], sim_seq[i]))

topn = 2

for i in range(100):
    print('\n Finding Top {} for "{}" '.format(topn, seqs[i]))

    find_topn_most_similar(seqs[i], topn)


