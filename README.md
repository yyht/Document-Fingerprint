# Authorship #
This project is built for **2018 IS Contest** in Wuhan, China.

In this project,
we want to generate a fingerprint for every article in Wechat Public Accounts,
with these fingerprints taking both the articles' topic information and writing style information
into account. 

The so-called document fingerprints can be used for Anomaly Detection, 
Similar Recommendation, Plagiarism Detection, etc. 


## Evironment ##
- Python 3.5
- Libraries: jieba, sklearn, keras (tensorflow backend), glove_python, lda, etc

## Data Explanation ##

We crawl data from **Wechat Public Accounts**. Although we have over 10000 articles for data mining,
only 150 of them are pushed. We cut the articles into stop words and topic words,
with python library jieba.

See /src/corpvir and /src/corpreal, 
here f means real words, g means virtual words.
Three accounts we push here is listed as follows:

- bd -- BigDataAbstract
- mm -- MiMeng 
- zjw -- Place for ZhangJiaWei's writing 

For Example in "bigdataAbstract":

- 0.txt ---- the original article of sequence 0 (50 altogether)
- 0bdf.txt ---- the real-word article of sequence 0
- 0bdg.txt ---- the virtual-word article of sequence 0
- punctuation.txt ---- stop words, used in cut.py

## Alogorithm Explanation ##

In brief, we utilize **LDA(Latent Dirichlet Allocation)** for Writing Style mining, 
**Glove-Tfidf** for Topic mining, and **VAE(Variational Auto-Encoder)** for Information Mixture. 
**Isolation Forest** is applied in the end for Anomaly Detection. 

We implement these algorithms in python. Here are some explanations for some critical .py files:

- cut.py: cut the original artical into stop word bags and topic word bags
- ldavir.py: LDA for the stop words
- glovereal.py: train Glove word embeddings
- tfidfreal.py: add Glove vectors of an article with their weights of TFIDF
- vae.py: Variational Auto-Encoder for information mixture
- iforest.py: Isolation forest for Anomaly Detection