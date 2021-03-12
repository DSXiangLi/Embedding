# -*- coding=utf-8 -*-

import pickle
import numpy as np
from encoder_decodeer_helper.tools import normalize


def find_topn_most_similar(seq_number, all_sentences, embedding_mat,  topn):
    emb = embedding_mat[seq_number, :]

    sim = np.matmul(np.expand_dims(emb, axis=0), embedding_mat.transpose())

    items = np.squeeze(np.argsort(sim, axis=1)[:, (-topn-1):-1], axis=0)

    scores = np.squeeze(sim[:, items], axis=0)

    sim_seq = all_sentences[items]


    for i in range(topn):
        print('Most Similar score = {:.2f} sentences = {}'.format(scores[i], sim_seq[i]))


def main(args):
    with open('./data/{}/{}_predict.pkl'.format(args.data, args.model), 'rb') as f:
        res = pickle.load(f)

    embedding_mat = np.array([item['vector'] for item in res])
    embedding_mat = np.apply_along_axis(normalize, 1, embedding_mat)

    seqs = np.array([' '.join([i.decode('utf-8') for i in item['input_token']]) for item in res])

    for i in range(args.num):
        print('\n Finding Top {} for "{}" '.format(args.topn, seqs[i]))
        find_topn_most_similar(i, seqs, embedding_mat, args.topn)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='[skip_thought|quick_thought|cnn_lstm]',
                        required=True)
    parser.add_argument('--data', type=str, help='train dataset',
                        required=False, default='bookcorpus')
    parser.add_argument('--topn', type=int, help='topN similar item to display',
                        required=False, default=1)
    parser.add_argument('--num', type=int, help='number of item to display',
                        required=False, default=50)
    args = parser.parse_args()
    main(args)