import jieba_fast as jieba
from tqdm import tqdm
import string
import re
import collections
import pickle
import itertools
import time
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp

class StrUtils(object):
    def __init__(self, path, language):
        self.puncts_file = '{}/puncts.txt'.format(path)
        self.stopwords_file = '{}/stop_words.txt'.format(path)
        self.re_dup_spaces = re.compile(r'(\s)(\1+)')
        self.re_content = re.compile(r'\<\/?[a-z]*\>')
        self.language = language

    @staticmethod
    def readline(file, lines):
        with open( file ) as f:
            for line in f:
                if len( line.strip() ) == 0:
                    continue
                lines.append( line.strip() )
        return lines

    @property
    def stop_words(self):
        return StrUtils.readline(self.stopwords_file, [])

    @property
    def re_puncts(self):
        # chinese puncts
        puncts = StrUtils.readline(self.puncts_file, [])
        # english puncts
        for p in list(string.punctuation):
            if p !='.':
                puncts.append('\\' +p)

        return re.compile(r'{}'.format('|'.join(puncts)))

    def text_cleaning(self, sentences):
        new_sentences = []
        for line in tqdm(sentences):
            line = self.re_content.sub( "", line )
            line = self.re_puncts.sub( "", line )
            line = self.re_dup_spaces.sub( "", line )
            new_sentences.append(line)

        return new_sentences

    def word_cut(self, sentences):
        if self.language == 'ch':
            func = lambda line: [i.strip() for i in jieba.cut(line, cut_all =False)]
        else:
            func = lambda line: line.split(" ")

        ##TODO: remove stop words or mark stop words
        t0 = time.time()
        word_cut = []
        for line in tqdm(sentences):
            try:
                words = func(line)
                words = [i for i in words if ((not i.isdigit()) and (i not in self.stop_words )) ]
                word_cut.append(words)
            except Exception as e:
                print(line)
                print(e)
                continue
        print('Single Process time {:.0f}'.format(time.time()-t0))
        return word_cut


    def multi_word_cut(self, sentences):
        print('Multiprocessing Word cut ')
        if self.language == 'ch':
            def func(line):
                line =  [i.strip() for i in jieba.cut(line, cut_all =False)]
                return [i for i in line if ((not i.isdigit()) and (i not in self.stop_words )) ]
        else:
            def func(line):
                return [i for i in line.split(" ") if ((not i.isdigit()) and (i not in self.stop_words))]

        pool = Pool(processes = int(mp.cpu_count()*0.8))
        t0 = time.time()
        word_cut = pool.map(func, sentences)
        print('MultiProcess  time {:.0f}'.format(time.time() - t0))
        return word_cut

    def make_ngram(self, ngram, sentences):
        print('Making ngram = {}'.format(ngram))
        if ngram == 1:
            return sentences
        else:
            tokens = []
            for line in tqdm(sentences):
                    token = [line[i:i+ngram] for i in range((len(line)-ngram) )]
                    token = [" ".join(i) for i in token]
                    tokens.append(token)
            return tokens


def dump_dictionary(output_path, sentences, prefix = ''):
    dict = collections.Counter(itertools.chain(*sentences))
    with open('{}/{}dictionary.pkl'.format(output_path, prefix), 'wb') as f:
        pickle.dump(dict, f )
