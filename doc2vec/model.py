from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
from tqdm import tqdm

import numpy as np

from doc2vec.gensim_utils import Progress, InputIter,  topn_similar_calc


class BaseModel(object):
    def __init__(self, params, model_dir, data_path):
        self.params = params
        self.model_dir = model_dir
        self.data_path = data_path
        self.model = None
        self.iterator = None
        self.init()

    def init(self):
        self.iterator = InputIter(self.data_path, tagged = self.params['tagged'])

    def fit(self, continue_train):
        raise NotImplementedError()

    def doc2vec(self, *kwargs):
        raise NotImplementedError()

    def model_save(self):
        self.model.save(self.model_dir)

    def model_load(self):
        raise NotImplementedError()


class Doc2VecModel( BaseModel ):
    def __init__(self, params, model_dir, data_path):
        super( Doc2VecModel, self ).__init__( params, model_dir, data_path )

    def fit(self, continue_train=False):
        """
        If continue train: load trained model, update vocabulary and continue training
        else: initialize model and train
        """
        if not continue_train:
            self.model = Doc2Vec( dm=self.params['dm'],  # dm=1 PV-DM, otherwise PV-DBOW
                                  vector_size=self.params['vector_size'],
                                  window_size=self.params['window_size'],
                                  min_count=self.params['min_count'],
                                  seed=self.params['seed'],
                                  sample=self.params['sample'],
                                  hs=self.params['hierarchy_softmax'],
                                  negative=self.params['negative_sampling'],
                                  workers=self.params['workers'],
                                  dm_mean=self.params['dm_mean'],  # avg/sum pooling of context
                                  dm_concat=self.params['dm_concat'],  # avergae/concat context
                                  dm_tag_count=self.params['dm_tag_count'],  # number of document tags
                                  dbow_words=self.params['dbow_words']  # train word vector along with doc vector
                                  )
        self.model.build_vocab(
            documents=self.iterator, update=continue_train
        )

        self.model.train(
            documents=self.iterator,
            total_examples=self.model.corpus_count,
            epochs=self.params['epochs'],
            start_alpha=self.params['alpha'],
            end_alpha=self.params['min_alpha'],
            callbacks=[Progress( 'Doc2Vec' )]
        )

    def doc2vec(self, tag=None, tokens=None):
        """
        Infer Document vector for train sample
        Input
            tag is None, return all vectors
            tag is not None, return insample corresponding embedding
            tokens is not None, infer OOB embeddinng
        Return
            embedding
        """
        if tag:
            return np.array( self.model[tag] )  # Doc2vec model is callable to get vector directly
        elif tokens:
            return self.model.infer_vector( tokens,
                                            alpha=self.params['alpha'],
                                            min_alpha=self.params['min_alpha'],
                                            epochs=self.params['infer_epochs'] )
        else:
            return self.model.docvecs.vectors_docs

    def most_similar_doc(self, tag=None, tokens=None, topn=10):
        """
        Calculate topn similar doc
        if tag is not none, use gensim function directly, vector is in sample
        if tokens is not none, infer oob vector and calculate similarity
        """
        if tag:
            items, score = self._most_similar_doc_insample( tag, topn )
            print( 'Top {} similar Documents for Doc {}'.format( topn, self.iterator[tag] ) )
        if tokens:
            items, score = self._most_similar_doc_oob( tokens, topn )
            print( 'Top {} similar Documents for Doc {}'.format( topn, ' '.join(tokens)) )

        sentences = self.iterator[items]

        for i in range( topn ):
            print( '{} score = {:.2f}: {}'.format( items[i], score[i], sentences[i] ) )

    def _most_similar_doc_insample(self, tag, topn):
        sim = self.model.docvecs.most_similar( positive=tag, topn=topn )

        items = [i[0] for i in sim]
        score = [i[1] for i in sim]
        return items, score

    def _most_similar_doc_oob(self, tokens, topn):
        insample_embedding = self.doc2vec()
        token_embedding = self.doc2vec( tokens=tokens )

        items, score = topn_similar_calc(insample_embedding ,token_embedding, topn)

        return items, score

    def model_load(self):
        self.model = Doc2Vec.load( self.model_dir )


class Word2VecModel( BaseModel ):
    def __init__(self, params, model_dir, data_path):
        super( Word2VecModel, self ).__init__( params, model_dir, data_path )
        self._train_embedding = None 

    def fit(self, continue_train):
        """
        If continue train: load trained model, update vocabulary and continue training
        else: initialize model and train
        """
        if not continue_train:
            self.model = Word2Vec( size=self.params['vector_size'],
                                   window=self.params['window_size'],
                                   min_count=self.params['min_count'],
                                   seed=self.params['seed'],
                                   hs=self.params['hierarchy_softmax'],
                                   negative=self.params['negative_sampling'],
                                   workers=self.params['workers'],
                                   cbow_mean=self.params['cbow_mean']  # 0 sum of context, 1 average of context
                                   )
        self.model.build_vocab(
            sentences=self.iterator, update=continue_train
        )

        self.model.train(
            sentences=self.iterator,
            total_examples=self.model.corpus_count,
            total_words=self.model.corpus_total_words,
            epochs=self.params['epochs'],
            start_alpha=self.params['alpha'],
            end_alpha=self.params['min_alpha'],
            callbacks=[Progress( 'Word2Vec' )]
        )

    def doc2vec(self, tag=None, tokens=None):
        """
        Use central embedding of word2vec as document vector
        Input
            tag is not None, find insample document, calculate central embedding
            tokens is not None, calcuulate central embedding
            both None, return all insample document vector. Will be cached after first called.
        Return
            docuemnt vector of tag or all
        """
        if tag:
            tokens = self.iterator[tag][0]
            embedding = self.central_embedding( tokens )
        elif tokens:
            embedding = self.central_embedding( tokens )
        else:
            if self._train_embedding is None :
                with tqdm( total=len( self.iterator ) ) as progress_bar:
                    embedding = []
                    for tokens in self.iterator:
                        embedding.append( self.central_embedding(tokens) )
                        progress_bar.update( 1 )
                    embedding = np.array( embedding )
                self._train_embedding = embedding
            else:
                return self._train_embedding
        return embedding

    def word2vec(self, word):
        try:
            embedding = self.model.wv[word]
        except KeyError:
            embedding = np.zeros( shape=(self.params['vector_size']) )
        return embedding

    def central_embedding(self, tokens):
        """
        Central embedding of sentences tokens
        """
        return np.average( [self.word2vec( i ) for i in tokens], axis=0 )

    def most_similar_doc(self, tag=None, tokens=None, topn=10):
        insample_embedding = self.doc2vec()
        token_embedding =  np.reshape(self.doc2vec( tag, tokens ), [1,-1])

        items, score = topn_similar_calc( insample_embedding, token_embedding, topn )
        sentences = self.iterator[items]

        if tag:
            tokens = self.iterator[tag]
        print( 'Top {} similar Documents for Doc {}'.format( topn, ' '.join( tokens ) ) )

        for i in range( topn ):
            print( '{} score = {:.2f}: {}'.format( items[i], score[i], sentences[i] ) )

    def model_load(self):
        self.model = Word2Vec.load( self.model_dir )
