# Authorship
For IS Contest

Data Explanation:
f means real words, g means virtual words
bd--BigDataAbstract    mm--MiMeng   zjw--ZhangJiaWei (three WeChats)
in "bigdata":
0.txt ---- the original article of sequence 0 (50 altogether)
0bdf.txt ---- the real-word article of sequence 0
0bdg.txt ---- the virtual-word article of sequence 0
punctuation.txt ---- stop words, used in cut.py 

Code Explanation:
ldareal.py ---- lda for all xxxf.txt
ldavir.py ---- lda for all xxxg.txt
cut.py ---- cut the original txt into xxxf.txt and xxxg.txt
hCluster.py ---- the library for cluster
