
import os
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
    encoder_path = '{}/encoder_source.txt'.format( data_dir )
    decoder_path = '{}/decoder_source.txt'.format(data_dir)

    preprocess = StrUtils(os.path.join(const_dir, language), language)

    print('Reading Raw corpus in {}'.format(input_path))
    sentences = preprocess.readline(input_path)

    print('String Preprocessing and word Segmentation')
    sentences = preprocess.text_cleaning(sentences)
    sentences = preprocess.multi_word_cut(sentences)

    print('Making Triplets out of clean corpus')
    train_sample = make_triplet(sentences)

    print('Writing triplets into encoder and decoder source at'.format(data_dir))
    with open(encoder_path, 'w', encoding='utf-8') as fe, open(decoder_path, 'w', encoding='utf-8') as fd :
        for encoder_source, decoder_source in train_sample:
            fe.write(' '.join(encoder_source).lower())
            fe.write('\n')
            fd.write(' '.join(decoder_source).lower())
            fd.write('\n')

    print( 'Dumping Original Dictionary' )
    dump_dictionary( data_dir, sentences, debug= True)


if __name__ == '__main__':
    main( 'data/bookcorpus', 'const', 'en' )

