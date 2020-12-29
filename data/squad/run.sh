#!/usr/bin/env bash
DIR='./data/squad'

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -P $DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -P $DIR

python $DIR/data_preprocess.py