# -*-coding:utf-8 -*-
import pickle

def pair_check(topn):
    with open('./data/wmt/transformer_predict.pkl', 'rb') as f:
        translate = pickle.load(f)

    for item in translate[:topn]:
        source = ' '.join([i.decode('utf-8') for i in item['input_token']])
        target = ''.join([i.decode('utf-8') for i in item['predict_vocab']])
        print('{} -> {}\n'.format(source, target))

if __name__ == '__main__':

    topn = 100
    pair_check(topn)

