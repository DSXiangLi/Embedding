#!/usr/bin/env bash
DIR='./data/wmt'

wget http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz -p $DIR
wget http://data.statmt.org/wmt17/translation-task/dev.tgz -p $DIR
tar -zvxf $DIR/training-parallel-nc-v12.tgz --directory $DIR
tar -zvxf $DIR/dev.tgz --directory $DIR

cp $DIR/training/*zh-en* $DIR/
cp $DIR/dev/*enzh* $DIR/

python $DIR/data_preprocess.py
