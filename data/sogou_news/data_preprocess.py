#import sys
#print( sys.path )
import os
from data.preprocess_util import *

def main(data_dir, const_dir, language):
    input_path = '{}/corpus.txt'.format( data_dir )
    output_path = '{}/corpus_new.txt'.format( data_dir )

    preprocess = StrUtils(os.path.join(const_dir, language), language)

    print('Reading Raw corpus in {}'.format(input_path))
    sentences = preprocess.readline(input_path)

    print('String Preprocessing and word Segmentation')
    sentences = preprocess.text_cleaning(sentences)
    sentences = preprocess.word_cut(sentences)

    print('Writing copurs in {}'.format(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sentences:
            f.write(' '.join(line))
            f.write('\n')

    print('Dumping Original Dictionary')
    dump_dictionary(data_dir, sentences)


if __name__ == '__main__':
    main('data/sogou_news', 'const', 'ch')