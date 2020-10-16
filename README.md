# Embedding

### Model Draft

#### 1. Word2vec demo 
bash data/sogou_news_big/extract_file.sh

python word2vec/main.py --model SG --train_algo HS  [Skip-gram Hierarchy Softmax]

#### 2. fasttext classification demo
python data/quora_fasttext/data_preprocess.py

python fasttext/main.py --gpu 1 [allow using gpu in tf.estimator]

#### 3. Doc2vec Demo and Comparison with Word2vec
bash data/sogou_news_big/extract_file.sh

bash doc2vec/model_run.sh

Comparison: doc2vec/doc2vec_vs_word2vec_sogou.ipynb


### Paper List 
#### 词向量
- [Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)
- [Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)
- [Word2Vec] word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method (2014)
- [Word2Vec] word2vec Parameter Learning Explained (2016)
- [Fasttext] Enriching Word Vectors with Subword Information (Facebook 2017)
- [Fasttext] [Fasttext]Bag of Tricks for Efficient Text Classification (Facebook 2016)
- [Glove] Global Vectors for Word Representation (2014)
- [ELMo] Deep contextualized word representations (2018)
#### 通用文本向量
- [Doc2vec] Distributed Representations of Sentences and Documents (Google 2014)
- [Doc2vec] A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SEN- TENCE EMBEDDINGS (2017)
- [Skip-Thought] Skip-Thought Vectors (2015)
- [Quick-Thought] Quick-Thought: AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS (Google 2018)
- [USE] Universal Sentence Encoder (Google 2018)
- [ULMFit] Universal Language Model Fine-tuning for Text Classification (fastai 2018)
- [GPT] Improving Language Understanding by Generative Pre-Training (openai 2018)
- [Bert] Well Read Students Learn Better On the Importance of Pretraining Compact Models（Google 2019)
- [Sentence-BERT] Sentence Embeddings using Siamese BERT-Networks (2019)
#### 物品向量
- [Item2Vec] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)
- [Airbnb] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)
- [DeepWalk] DeepWalk- Online Learning of Social Representations (SBU 2014)
- [Node2vec] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)
- [Alibaba] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (Alibaba 2018)
#### 召回
- [LSH] Locality-Sensitive Hashing for Finding Nearest Neighbors (2008)

### Blog 
- 无所不能的Embedding1 - Word2vec模型详解&代码实现 https://www.cnblogs.com/gogoSandy/p/13418257.html
- 无所不能的Embedding2 - FastText词向量&文本分类 https://www.cnblogs.com/gogoSandy/p/13618077.html
- 无所不能的Embedding3 - word2vec->Doc2vec[PV-DM/PV-DBOW] https://www.cnblogs.com/gogoSandy/p/13773327.html
