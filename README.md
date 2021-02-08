# Embedding
Requirement: Tensorflow 1.14, Tensorboard 1.14

### Model Draft
#### 1. Word2vec Demo 
bash data/sogou_news_big/extract_file.sh

python word2vec/main.py --model SG --train_algo HS  [Skip-gram Hierarchy Softmax]

#### 2. fasttext classification Demo
python data/quora_fasttext/data_preprocess.py

python fasttext/main.py --gpu 1 [allow using gpu in tf.estimator]

#### 3. Doc2vec Demo and Comparison with Word2vec
bash data/sogou_news_big/extract_file.sh

bash doc2vec/model_run.sh

Comparison: doc2vec/doc2vec_vs_word2vec_sogou.ipynb

#### 4. Encoder-Decoder Family
数据： bash data/bookcorpus/run.sh

预训练词向量： bash data/pretrain_model/download_pretrain.sh

老代码在skip_thought_archived, 重构代码如下

skip-thought:  python skip_thought/main.py --clear_model 1 --step train --gpu 1

cnn-lstm:  python cnn_lstm/main.py --clear_model 1 --step train --gpu 1

#### 5. Transformer 
bash data/wmt/run.sh 

python transformer/main.py --clear_model 1 --step train --gpu 1 

### Paper List 
#### 词向量
- [Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)
- [Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)
- [Word2Vec] word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method (2014)
- [Word2Vec] word2vec Parameter Learning Explained (2016)
- [Fasttext] Enriching Word Vectors with Subword Information (Facebook 2017)
- [Fasttext] Bag of Tricks for Efficient Text Classification (Facebook 2016)
- [Glove] Global Vectors for Word Representation (2014)
- [ELMo] Deep contextualized word representations (2018)
#### 文本向量
- [Doc2vec] Distributed Representations of Sentences and Documents (Google 2014)
- [Doc2vec] A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SEN- TENCE EMBEDDINGS (2017)
- [Encoder-Decoder: Skip-Thought] Skip-Thought Vectors (2015)
- [Encoder-Decoder: Skip-Thought] Rethinking Skip-thought- A Neighborhood based Approach (2017)
- [Encoder-Decoder: CNN-LSTM]Learning Generic Sentence Representations Using Convolutional Neural Networks (2017)
- [Encoder-Decoder: Quick-Thought] Quick-Thought: AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS (Google 2018)
- [Transformer] Attention is all you need (2017)
- [FastSent|DVAE]Learning Distributed Representations of Sentences from Unlabelled Data (2016)
- [Siamese] Learning Text Similarity with Siamese Recurrent Networks (2016)
- [InferSent] Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (2018)
- [GenSen] LEARNING GENERAL PURPOSE DISTRIBUTED SENTENCE REPRESENTATIONS VIA LARGE SCALE MULTITASK LEARNING (2018)
- [USE] Universal Sentence Encoder (Google 2018)
- [ULMFit] Universal Language Model Fine-tuning for Text Classification (fastai 2018)
- [GPT] Improving Language Understanding by Generative Pre-Training (openai 2018)
- [Bert] Pre-training of Deep Bidirectional Transformers for Language Understanding（Google 2019)
- [Sentence-BERT] Sentence Embeddings using Siamese BERT-Networks (2019)
- [Bert-flow] On the Sentence Embedding from Pre-trained Language Model(2020)
- [Representation] Fine-Grained Analysis of Sentence Embedding Using Auxiliary Prediction Tasks (2017)
- [Representation] What you can cram into a single vector: Probing Sentence Embedding for linguistic properties(2018)
- [Representation] Assessing Composition in Sentence Vector Representations (2018)

#### 物品向量
- [Item2Vec] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)
- [Airbnb] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)
- [DeepWalk] DeepWalk- Online Learning of Social Representations (SBU 2014)
- [Node2vec] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)
- [Alibaba] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (Alibaba 2018)
#### 召回
- [LSH] Locality-Sensitive Hashing for Finding Nearest Neighbors (2008)
- [HNSW] Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs(2016)

### Blog 
- [无所不能的Embedding1 - Word2vec模型详解&代码实现](https://www.cnblogs.com/gogoSandy/p/13418257.html)
- [无所不能的Embedding2 - FastText词向量&文本分类](https://www.cnblogs.com/gogoSandy/p/13618077.html)
- [无所不能的Embedding3 - word2vec->Doc2vec[PV-DM/PV-DBOW]](https://www.cnblogs.com/gogoSandy/p/13773327.html)
- [无所不能的Embedding4 - Doc2vec第二弹[skip-thought & tf-Seq2Seq源码解析] ](https://www.cnblogs.com/gogoSandy/p/14009265.html)
- [无所不能的Embedding5 - skip-thought的兄弟们[Trim/CNN-LSTM/quick-thought] ](https://www.cnblogs.com/gogoSandy/p/14094590.html)
- [无所不能的Embedding6 - 跨入Transformer时代～模型详解&代码实现](https://www.cnblogs.com/gogoSandy/p/14386998.html)
