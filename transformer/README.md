## Transformer 
paper: Attention is all you need 

这里我们用WMT英翻中来实现Transformer，加入预训练词向量来初始化Embedding

1. 下载并解析清洗数据，写训练集，测试集和词典

bash data/wmt/run.sh
   
2. 下载预训练词向量,英文是google-word2vec300, 中文是Fasttext300。也可以选择随机初始化Embedding，把config里面pretrain_model_list改成[]即可

bash data/pretrain_model/download_pretrain.sh 
   
3. 训练模型：发现transformer batch_size太小不收敛，所以模型参数给的比较小不然会报OOM

python transformer/main.py --gpu 1 --clear_model 1

<p align="left"> 
    <img src="https://static01.imgkr.com/temp/581eb15bc4ef486dbdd11acd501b7557.png" width="600" height="100"> </p> 
<p align="left"> </p> 

<p align="left"> 
    <img src="https://static01.imgkr.com/temp/d17644a667af408197f621dee6373653.png" width="600" height="100"> </p> 
<p align="left"> </p> 

<p align="left"> 
    <img src="https://static01.imgkr.com/temp/b7d848c77382483a9096bab343ebb91b.png" width="300" height="200"> </p> 
<p align="left"> </p> 

4. 预测

python transformer/main.py --step predict --gpu 1 --clear_model 1 

case: test_inference.py 
