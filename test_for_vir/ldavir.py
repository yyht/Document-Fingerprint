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
    for i in range(0,120):
        if (i % 3 == 0):
            f = open("bd"+ str(i//3)+"g.txt",'r',encoding= 'utf-8')
        elif (i % 3 == 1):
            f = open("mm" + str(i//3) + "g.txt", 'r', encoding='utf-8')
        else :
            f = open("zjw" + str(i//3) + "g.txt", 'r', encoding='utf-8')
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

    # 输出主题的分布
    label2 = []
    n_top_words = 7
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    n_weight = np.zeros((30, len(vc)))
    n_corpus = []
    vcorigin = list(vectorizer.get_feature_names())
    for i in range(40, 50):
        titles = ["bd","mm","zjw"]
        for j in range(len(titles)):
            f = open(titles[j] + str(i) + "g.txt", 'r', encoding='utf-8')
            tmp = f.read()
            n_corpus.append(tmp.split())
    lent = 0
    for i in range(len(n_corpus)):
        lent = 0
        for word in n_corpus[i]:
            for j in range(len(vcorigin)):
                if word == vcorigin[j]:
                    n_weight[i][j] += 1
                    lent += 1
                    break
        print((i+120), '  ', lent / len(n_corpus[i]))

    B = np.asarray(n_weight.astype(np.int32))
    C = model.transform(B, max_iter=500, tol=1e-16)
    outcome = np.concatenate((doc_topic, C))

    # 输出文章的主题分布
    label = []
    for n in range(len(outcome)):
        label.append(outcome[n])

    k, l = hcluster(outcome, 10)
    print(l)
    for subl in l:
        subl.sort()
    for i in range(len(l)):
        if len(l[i]) >= 1:
            for j in range(len(l[i])):
                print("type: {} doc: {} topic: {}".format( (l[i][j] % 3),l[i][j], label[l[i][j]]))
        print('\n')


