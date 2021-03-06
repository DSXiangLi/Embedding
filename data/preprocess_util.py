import jieba_fast as jieba
from tqdm import tqdm
import string
import re
import collections
import pickle
import itertools
import time
from pathos.multiprocessing import ProcessingPool as Pool


class StrUtils(object):
    def __init__(self, path, language):
        self.puncts_file = '{}/puncts.txt'.format(path)
        self.stopwords_file = '{}/stop_words.txt'.format(path)
        self.re_dup_spaces = re.compile(r'(\s)(\1+)')
        self.re_content = re.compile(r'\<\/?[a-z]*\>')
        self.language = language

    @staticmethod
    def readline(file):
        lines = []
        with open( file ,encoding='UTF-8') as f:
            for line in f:
                if len( line.strip() ) == 0:
                    continue
                lines.append( line.strip().lower() )
        return lines

    @property
    def stop_words(self):
        return StrUtils.readline(self.stopwords_file)

    @property
    def re_puncts(self):
        # chinese puncts
        puncts = StrUtils.readline(self.puncts_file)
        # english puncts
        for p in list(string.punctuation):
            if p !='.':
                puncts.append('\\' +p)

        return re.compile(r'{}'.format('|'.join(puncts)))

    def text_cleaning(self, sentences):
        new_sentences = []
        for line in tqdm(sentences):
            line = self.re_content.sub( "", line )
            line = self.re_puncts.sub( " ", line )
            line = self.re_dup_spaces.sub( " ", line )
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
                if self.language =='ch':
                    words = [i for i in words if ((not i.isdigit()) and (i not in self.stop_words ) ) ]
                else:
                    words = [i for i in words if ((not i.isdigit()) and (i not in self.stop_words ) and (len(i)>1) ) ]
                if len(words) >1:
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
            jieba.initialize()  # initialize first, or it will initialize in each process
            jieba.disable_parallel()

            def func(line):
                line =  [i.strip() for i in jieba.cut(line, cut_all =False)]
                return [i for i in line if ((not i.isdigit()) and (i not in self.stop_words )) ]
        else:
            def func(line):
                return [i.lower() for i in line.split(" ") if ((not i.isdigit()) and \
                                                       (i not in self.stop_words) and \
                                                       (len(i) >1 ) )]

        pool = Pool(nodes=5)
        t0 = time.time()
        word_cut = pool.map(func, sentences)
        pool.close()
        pool.join()
        pool.clear()
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


str_utils_en = StrUtils('./const/en','en')
str_utils_ch = StrUtils('./const/ch','ch')

def dump_dictionary(output_path, sentences, prefix = '', debug=False, dry_run=False):
    dict = collections.Counter(itertools.chain.from_iterable(sentences))
    if not dry_run:
        print('Dumping Original Dictionary')
        with open('{}/{}dictionary.pkl'.format(output_path, prefix), 'wb') as f:
            pickle.dump(dict, f )
    if debug:
        print('Dictionary size = {}'.format(len(dict)))
        print(list(dict.most_common(10)))

