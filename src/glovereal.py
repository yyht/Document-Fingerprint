import array

import pytest

import numpy as np
import scipy.sparse as sp

from glove import Corpus,Glove
from glove.glove import check_random_state

from utils import (build_coocurrence_matrix,
                   generate_training_corpus)

# read the real words corpus    
def readRealWords():
    corpus = []
    for i in range(0,120):
        if (i % 3 == 0):
            f = open("corpreal/bd"+ str(i//3)+"f.txt",'r',encoding= 'utf-8')
        elif (i % 3 == 1):
            f = open("corpreal/mm" + str(i//3) + "f.txt", 'r', encoding='utf-8')
        else :
            f = open("corpreal/zjw" + str(i//3) + "f.txt", 'r', encoding='utf-8')
        tmp = f.read()
        tmp2 = tmp.split()
        corpus.append(tmp2)
    return corpus

# generate Corpus object from realword corpus
# cooccurance matrix 
def genCorp(corpus,windowsize=5):
    corpusmodel = Corpus()
    corpusmodel.fit(corpus, window=windowsize)
    return corpusmodel

# generate glove_model from corpus model
def corpus_to_glove(corpus):
    glove_model = Glove(no_components=20, learning_rate=0.07)
    glove_model.fit(corpus.matrix,
                    epochs=1000,
                    no_threads=2)   
    glove_model.add_dictionary(corpus.dictionary)
    return glove_model

def Veccos(vector1,vector2):  
    dot_product = 0.0  
    normA = 0.0  
    normB = 0.0  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5)

if __name__ == "__main__":
    tag = 0
    if tag:    
        cop = readRealWords()
        corpmodel = genCorp(cop,5)
        glomodel = corpus_to_glove(corpmodel)
        glomodel.save('gloveSave.txt')
    else:
        glomodel = Glove.load('gloveSave.txt')
    try:
        word_idx1 = glomodel.dictionary['机器']
        word_idx2 = glomodel.dictionary['学习']
    except:
        pass
    v1 = glomodel.word_vectors[word_idx1]
    v2 = glomodel.word_vectors[word_idx2]
    print(Veccos(v1,v2))
    
    print(glomodel.most_similar('机器',20))
    print(glomodel.most_similar('学习',20)) 

    
    




