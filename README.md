# Embedding

### Model Draft

#### 1. Word2vec demo 
bash data/sogou_news_big/extract_file.sh

python word2vec/main.py --model SG --train_algo HS  [Skip-gram Hierarchy Softmax]

#### 2. fasttext classification demo
python data/quora_fasttext/data_preprocess.py

python fasttext/main.py --gpu 1 [allow using gpu in tf.estimator]

### Paper List 
- [Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)
- [Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)
- [Word2Vec] word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method (2014)
- [Word2Vec] word2vec Parameter Learning Explained (2016)
- [Fasttext] Enriching Word Vectors with Subword Information (Facebook 2017)
- [Fasttext] [Fasttext]Bag of Tricks for Efficient Text Classification (Facebook 2016)
- [Glove] Global Vectors for Word Representation (2014)
- [Doc2vec] Distributed Representations of Sentences and Documents (Google 2014)
- [Item2Vec] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)
- [DeepWalk] DeepWalk- Online Learning of Social Representations (SBU 2014)
- [Node2vec] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)
- [Airbnb] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)
- [Alibaba] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (Alibaba 2018)
- [LSH] Locality-Sensitive Hashing for Finding Nearest Neighbors (2008)

### Blog 
- 无所不能的Embedding 1 - Word2vec模型详解&代码实现 https://www.cnblogs.com/gogoSandy/p/13418257.html
- 无所不能的Embedding 2. FastText词向量&文本分类 https://www.cnblogs.com/gogoSandy/p/13618077.html
