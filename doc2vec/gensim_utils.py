from glob import glob
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
import itertools
import numpy as np

class InputIter( object ):
    """
    Generate Input Iterable for gensim doc2vec/word2vec model, in order to avoid loading entire corpus in memory
    Note. gensim only accept iterator not generator, so that it can stream over the data repeatedly
    """

    def __init__(self, data_path, tagged):
        self.data_path = data_path
        self.tagged = tagged
        self.param_check()

    def param_check(self):
        assert glob( self.data_path ), 'data file not found'

    def __iter__(self):
        with open( self.data_path, 'r' ) as f:
            for i, line in enumerate( f ):
                if self.tagged:
                    yield TaggedDocument( words=line.strip().split( " " ), tags=[i] )
                else:
                    yield line.strip().split( " " )

    def __getitem__(self, index):
        """
        allow single/list/slice item extraction
        """

        def slice_helper(start, stop, step, index = None):
            tokens = []
            with open( self.data_path, 'r' ) as f:
                for i, line in enumerate(itertools.islice(f, start, stop, step)):
                    if index is None:
                        tokens.append( line.strip().split(' ') )
                    else:
                        if (start+i) in index:
                            tokens.append( line.strip().split(' '))
            return tokens

        if isinstance(index, int):
            tokens = slice_helper( start=index, stop=(index+1), step=1, index=None )

        elif isinstance(index, list):
            tokens = slice_helper( start=min(index), stop=(max(index)+1), step=1, index= index )

        elif isinstance( index, slice):
            tokens = slice_helper(start = index.start, stop = index.stop, step = index.step, index = None)

        else:
            raise Exception('Only int, list, slice type are supported')

        return tokens

    def __len__(self):
        with open( self.data_path, 'r' ) as f:
            count = len(f.readlines())
        return count


class Progress( CallbackAny2Vec ):
    def __init__(self, model_name):
        self.epoch = 0
        self.model_name = model_name

    def on_train_begin(self, model):
        print( 'Start training {} Model ...\n'.format( self.model_name ) )

    def on_epoch_end(self, model):
        print( ("=" * 5 + "Epoch {} Finished" + "=" * 5).format( self.epoch ) )
        self.epoch += 1


def normalize(vector):
    norm = np.linalg.norm( vector )
    if norm == 0:
        norm = np.finfo( vector.dtype ).eps
    return vector / norm


def topn_similar_calc(mat1, mat2, topn):
    """
    matrix: n_doc * emb_size
    """
    score = np.matmul( np.apply_along_axis( normalize, 1, mat1 ),
                       np.apply_along_axis( normalize, 1, mat2 ).transpose())
    score = np.squeeze(score)
    items = np.argsort(score)[::-1][:topn]
    score = score[items]

    return list( items ), score
