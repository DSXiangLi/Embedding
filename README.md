# Embedding

### Model Draft

#### 1. Word2vec demo 
bash data/sogou_news_big/extract_file.sh

python data/sogou_news_big/data_preprocess.py

python word2vec/main.py --model SG --train_algo HS  [Skip-gram Hierarchy Softmax]


### Paper List 
- [Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)
- [Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)
- [Doc2vec] Distributed Representations of Sentences and Documents (Google 2014)
- [Glove] Global Vectors for Word Representation (2014)
- [Fasttext] Enriching Word Vectors with Subword Information (Facebook 2017)
- [Item2Vec] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)
- [DeepWalk] DeepWalk- Online Learning of Social Representations (SBU 2014)
- [Node2vec] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)


### Blog 
- 无所不能的Embedding 1 - Word2vec模型详解&代码实现 https://www.cnblogs.com/gogoSandy/p/13418257.html