#!/usr/bin/env bash

python doc2vec/main.py --model doc2vec --continue_train 0 --data sogou_news_big
python doc2vec/main.py --model word2vec --continue_train 0  --data sogou_news_big
