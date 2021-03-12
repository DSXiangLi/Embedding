## Word2Vec from scratch 
Tutorial: [无所不能的Embedding1 - Word2vec模型详解&代码实现](https://www.cnblogs.com/gogoSandy/p/13418257.html)

1. 下载sogou新闻数据，模型defualt用小数据

bash data/sogou_news_big/extract_file.sh

bash data/sogou_news/extract_file.sh

2. 训练: 
   
模型[--model]选择支持skip-gram[SG]，continuous bag of word[CBOW]

训练[--train_algo]支持Hierarchy Softmax[HS], 和negative sampling[NG]

其中negative sampling对应的[--loss]可以是nce_loss和sample_loss

python word2vec/main.py --model SG --train_algo HS