

import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data.preprocess_util import *


def make_triplet(sentences):
    """
    Split all sentences into triplets : (s_{i-1}, s_i, s_{i+1})
    out of which 2 pair of train samples (s_i, s_{i-1}), (s_i, s_{i+1}) will be created.
    s_i will be the input of encoder
    s_{i-1}, s_{i+1} will be the input of decoder
    """
    train_sample = []
    for i in range(1, (len(sentences)-1) ):
        train_sample.append( (sentences[i], sentences[i - 1]) )
        train_sample.append( (sentences[i], sentences[i + 1]) )

    return train_sample


def main(data_dir, const_dir, language):
    input_path = '{}/all.tokenized.txt'.format( data_dir )
    train_encoder_path = '{}/train_encoder_source.txt'.format( data_dir )
    train_decoder_path = '{}/train_decoder_source.txt'.format(data_dir)

    dev_path = '{}/dev_encoder_source.txt'.format( data_dir )

    preprocess = StrUtils(os.path.join(const_dir, language), language)

    print('Reading Raw corpus in {}'.format(input_path))
    sentences = preprocess.readline(input_path)

    print('String Preprocessing and word Segmentation')
    sentences = preprocess.text_cleaning(sentences)
    sentences = preprocess.multi_word_cut(sentences)

    with open('{}/all_sentences.txt'.format(data_dir), 'w', encoding='utf-8') as f:
        for line in sentences:
            f.write(' '.join(line).lower())
            f.write('\n')

    print('Making Triplets out of clean corpus')
    train, dev = train_test_split(sentences, test_size=0.2, random_state=1234)
    train = make_triplet(train)
    train = shuffle(train)

    print('Writing triplets into encoder and decoder source at {}'.format(data_dir))
    with open(train_encoder_path, 'w', encoding='utf-8') as fe, open(train_decoder_path, 'w', encoding='utf-8') as fd :
        for encoder_source, decoder_source in train:
            fe.write(' '.join(encoder_source).lower())
            fe.write('\n')
            fd.write(' '.join(decoder_source).lower())
            fd.write('\n')

    with open(dev_path, 'w', encoding='utf-8') as fe:
        for text in dev:
            fe.write(' '.join(text).lower())
            fe.write('\n')

    print( 'Dumping Original Dictionary' )
    dump_dictionary( data_dir, sentences, debug= True)


if __name__ == '__main__':
    data_dir = 'data/bookcorpus'
    const_dir = 'const'
    language = 'en'
    main( data_dir, const_dir, language)