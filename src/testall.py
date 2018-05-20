import array

import pytest

import numpy as np
import scipy.sparse as sp

from glove import Corpus,Glove
from glove.glove import check_random_state

from utils import (build_coocurrence_matrix,
                   generate_training_corpus)

def test_corpus_construction():

    corpus_words = ['a', 'naïve', 'fox','a','a','fox','zhr','laji']
    corpus_words2 = ['a', 'sillyb', 'fox','fox','a','fox','zjl','zhr','uyu','a','fox','a']
    corpus = [corpus_words,corpus_words2]
    #print(corpus)   

    model = Corpus()
    model.fit(corpus, window=10)

    #print("model.matrix.shape",model.matrix.shape)
    #print(model.dictionary) 

    '''expected = [[0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]]'''

    #print("model.matrix.todense().tolist()",model.matrix.todense().tolist())
    return model

'''
def _reproduce_input_matrix(glove_model):

    wvec = glove_model.word_vectors
    #print(wvec)
    wbias = glove_model.word_biases

    out = np.dot(wvec, wvec.T)

    for i in range(wvec.shape[0]):
        for j in range(wvec.shape[0]):
            if i == j:
                out[i, j] = 0.0
            elif i < j:
                out[i, j] += wbias[i] + wbias[j]
            else:
                out[i, j] = 0.0

    return np.asarray(out)
'''

def corpus_to_glove(corpus):
    glove_model = Glove(no_components=100, learning_rate=0.05)
    glove_model.fit(corpus.matrix,
                    epochs=500,
                    no_threads=2)
    glove_model.add_dictionary(corpus.dictionary)
    return glove_model

corpusmodel = test_corpus_construction()
#print(corpusmodel.matrix)
glovemodel = corpus_to_glove(corpusmodel)
print(glovemodel.most_similar('a', number=3))
