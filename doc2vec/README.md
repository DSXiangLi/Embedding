## 对比Word2vec和Doc2Vec在文本表达的差异[PV-DBOW/PV-DM]
Tutorial: [无所不能的Embedding3 - word2vec->Doc2vec[PV-DM/PV-DBOW]](https://www.cnblogs.com/gogoSandy/p/13773327.html)

1. 下载sogou新闻数据，模型defualt用小数据

bash data/sogou_news_big/extract_file.sh

bash data/sogou_news/extract_file.sh

2. 训练Doc2Vec模型, 这里用了gensim

bash doc2vec/model_run.sh

3. 对比词向量和文本向量，在semantic similarity上的差异

doc2vec/doc2vec_vs_word2vec_sogou.ipynb