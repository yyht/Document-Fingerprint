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
from Hcluster import hcluster

if __name__ == "__main__":

    # 存储读取语料 一行预料为一个文档
    corpus = []
    for i in range(0,80):
        if (i % 2 == 0):
            f = open("bd"+ str(i//2)+"f.txt",'r',encoding= 'utf-8')
        elif (i % 2 == 1):
            f = open("mm" + str(i//2) + "f.txt", 'r', encoding='utf-8')

        tmp = f.read()
        corpus.append(tmp)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()
    vc = list(vectorizer.get_feature_names())
    for i in range(len(vc)):
        if vc[i][0:2] == "填充":
            vc[i] = vc[i][2:]
    vocab = tuple(vc)
    #print(len(vectorizer.vocabulary_))
    #print(len(lda.datasets.load_reuters_vocab()))

    model = lda.LDA(n_topics=6, n_iter=500, random_state=1)
    model.fit(np.asarray(weight.astype(np.int32)))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    # 输出文章的主题分布
    label = []
    for n in range(80):
        #topic_most_pr = doc_topic[n].argmax()
        label.append(doc_topic[n])
        #print("doc: {} topic: {}".format(n, label[n]))

    # 输出主题的分布
    label2 = []
    n_top_words = 7
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    k,l = hcluster(doc_topic,10)
    for i in range(len(l)):
        if len(l[i]) >=1:
            for j in range(len(l[i])):
                print("doc: {} topic: {}".format(l[i][j], label[l[i][j]]))
        print('\n')