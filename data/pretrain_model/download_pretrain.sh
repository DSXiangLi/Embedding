#!/usr/bin/env bash
DIR='./data/pretrain_model'

wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -P $DIR
gunzip $DIR/GoogleNews-vectors-negative300.bin.gz

wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz" -P $DIR
gunzip $DIR/cc.zh.300.bin.gz
