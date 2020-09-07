#!/usr/bin/env bash
DATAPATH=data/sogou_news
tar -zvxf $DATAPATH/news_sohusite_xml.full.tar.gz
cat $DATAPATH/news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  > $DATAPATH/corpus.txt
python $DATAPATH/data_preprocess.py