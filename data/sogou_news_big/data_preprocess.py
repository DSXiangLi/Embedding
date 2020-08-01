import jieba_fast as jieba
from tqdm import tqdm
import string
import re

class StrUtils(object):
    def __init__(self, path = 'const'):
        self.puncts_file = '{}/puncts.txt'.format(path)
        self.stopwords_file = '{}/stop_words.txt'.format(path)
        self.re_dup_spaces = re.compile(r'(\s)(\1+)')
        self.re_content = re.compile(r'\<\/?[a-z]*\>')

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

    def preprocess(self, sentences):
        new_sentences = []
        for line in tqdm(sentences):
            line = self.re_content.sub( "", line )
            line = self.re_puncts.sub( "", line )
            line = self.re_dup_spaces.sub( "", line )
            new_sentences.append(line)

        return new_sentences

    def word_cut(self, sentences):
        word_cut = []
        for line in tqdm(sentences):
            try:
                words = [i.strip() for i in jieba.cut(line, cut_all= False)]
                words = [i for i in words if (not i.isdigit())\
                                     and (i not in self.stop_words ) \
                                     and (len(i)>1)]
                word_cut.append(words)
            except Exception as e:
                print(line)
                print(e)
                continue
        return word_cut


if __name__ == '__main__':
    data_dir = 'data/sogou_news_big'
    input_path = '{}/corpus.txt'.format( data_dir )
    output_path = '{}/corpus_new.txt'.format( data_dir )

    print('Reading Raw corpus in {}'.format(input_path))
    sentences = []
    with open(input_path, 'r' ) as f:
        for line in f:
            sentences.append( line )

    print('String Preprocessing and word Segmentation')
    jieba.enable_parallel(processnum = 8)
    preprocess = StrUtils()
    sentences = preprocess.preprocess(sentences)
    sentences = preprocess.word_cut(sentences)

    print('Writing copurs in {}'.format(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sentences:
            f.write(' '.join(line))
            f.write('\n')
