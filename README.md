# NLP Embedding
Requirement: Tensorflow 1.14, Tensorboard 1.14

The code is not rigorously tested, if you find a bug, welcome PR ^_^ ~

### Model List
1. Word2Vec: sogou新闻数据
2. Fasttext: quora kaggle 分类数据
3. Doc2Vec[PV-DBOW/PV-DM]: sogou新闻数据
4. skip-thought: bookcorpus爬虫数据
5. quick-thought: bookcorpus爬虫数据
6. CNN-LSTM: bookcorpus爬虫数据
7. transformer: WMT英翻中任务

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


### Blog 
- [无所不能的Embedding1 - Word2vec模型详解&代码实现](https://www.cnblogs.com/gogoSandy/p/13418257.html)
- [无所不能的Embedding2 - FastText词向量&文本分类](https://www.cnblogs.com/gogoSandy/p/13618077.html)
- [无所不能的Embedding3 - word2vec->Doc2vec[PV-DM/PV-DBOW]](https://www.cnblogs.com/gogoSandy/p/13773327.html)
- [无所不能的Embedding4 - skip-thought & tf-Seq2Seq源码解析](https://www.cnblogs.com/gogoSandy/p/14009265.html)
- [无所不能的Embedding5 - skip-thought的兄弟们[Trim/CNN-LSTM/quick-thought] ](https://www.cnblogs.com/gogoSandy/p/14094590.html)
- [无所不能的Embedding6 - 跨入Transformer时代～模型详解&代码实现](https://www.cnblogs.com/gogoSandy/p/14386998.html)
- [无所不能的Embedding7 - 探索通用文本表达[FastSent/InferSent/GenSen/USE] ](https://www.cnblogs.com/gogoSandy/p/14439293.html)
