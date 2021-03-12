## Fasttext 词向量和文本分类任务
Tutorial: [无所不能的Embedding2 - FastText词向量&文本分类](https://www.cnblogs.com/gogoSandy/p/13618077.html)

1. 下载Quora Insincere kaggle分类数据后做数据清洗

python data/quora_fasttext/data_preprocess.py

2. 数据EDA

fasttext/quora_insincerece_EDA.ipynb

3. 训练分类任务

python fasttext/main.py --gpu 1

