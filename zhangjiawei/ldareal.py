#coding=utf-8
import os
import numpy as np
import jieba
import scipy
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import lda
import sys
import lda.datasets


if __name__ == "__main__":

    # 存储读取语料 一行预料为一个文档   
    corpus = []
    for i in range(0,40):
        f = open(str(i)+"f.txt",'r',encoding= 'utf-8')
        tmp = f.read()
        corpus.append(tmp)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()
    vocab = tuple(vectorizer.get_feature_names())
    #print(len(vectorizer.vocabulary_))
    #print(len(lda.datasets.load_reuters_vocab()))

    model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model.fit(np.asarray(weight.astype(np.int32)))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    # 输出文章的主题分布
    label = []
    for n in range(40):
        #topic_most_pr = doc_topic[n].argmax()
        label.append(doc_topic[n])
        print("doc: {} topic: {}".format(n, label[n]))

    # 输出主题的分布
    label2 = []
    n_top_words = 5
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
