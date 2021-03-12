## CNN-LSTM
Tutorial: [无所不能的Embedding5 - skip-thought的兄弟们[Trim/CNN-LSTM/quick-thought] ](https://www.cnblogs.com/gogoSandy/p/14094590.html)

1. 下载解析并清洗数据，写训练集，测试集和词典，其中bookcorpus用了开源的一个爬虫项目要

git submodule add git@github.com:soskek/bookcorpus.git

bash data/bookcorpus/run.sh
   
2. 下载预训练词向量英文用的是google-word2vec300，如果选择随机初始化Embedding可以跳过这一步，把config里面pretrain_model_list改成 [] 即可

bash data/pretrain_model/download_pretrain.sh 
   
3. 训练模型

python cnn_lstm/main.py --gpu 1 --clear_model 1

4. 生成文本Embedding，并搜索TopN相似的文本

python cnn_lstm/main.py --step predict --clear_model 0 

TopN Search: python encoder_decoder_helper/most_similar_test.py --model cnn_lstm --topn 1 --num 50