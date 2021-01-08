#!/usr/bin/env bash
DIR='./data/pretrain_model'

wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -P $DIR
gunzip $DIR/GoogleNews-vectors-negative300.bin.gz
