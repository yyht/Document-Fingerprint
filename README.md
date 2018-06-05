# Document Fingerprint#
This project is built for **2018 IS Contest** in Wuhan, China.

In this project,
we aim at generating a fingerprint for every article in Wechat Public Accounts,
with these fingerprints taking both the articles' topic information and writing style information
into account. 

The so-called document fingerprints can be used for Anomaly Detection, 
Similar Recommendation and Plagiarism Detection, etc. 


## Environment ##
- Python 3.5
- Libraries: jieba, sklearn, keras (tensorflow backend), glove_python, lda, wordcloud, matplotlib, etc

## Data Explanation ##

We crawl data from **Wechat Public Accounts**. Although we have over 10000 articles for data mining,
only 1300 of them are pushed on github. We will cut the articles into stop words and topic words for further usage,
with python library jieba.

See \auth_server\fingerprint\data for data. The seven accounts we push here is listed as follows:

- gkw -- Guo Ke
- dsjwz -- Big Data Abstract
- mm -- Mi Meng
- xkd -- Xia Ke Island
- jqzx -- Machine's Heart, etc.

## Alogorithm Explanation ##

In brief, we utilize **LDA(Latent Dirichlet Allocation)** and **Synonym Forest** for Writing Style mining, 
**Glove-Tfidf** for Topic mining, and **VAE(Variational Auto-Encoder)** for Information Mixture. 
**Isolation Forest** is applied in the end for Anomaly Detection. We implement these algorithms in python.
Your can find some critical files in \training.

## Server and Client ##

We build a **Wechat mini program** to demonstrate our masterpiece, see \client for more details. 
As for the server, we apply **tornado** to construct the server, see \auth_server.

## More Information ##

In \doc, there is our ducument for this project, written in Chinese. The document consists of 
the detailed explantions of our algorithm, implementation and thoughts. Hope it will help you.

This project is done by WL, GJJ, HJC and me, at SJTU.