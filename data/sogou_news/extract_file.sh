#!/usr/bin/env bash
DATAPATH=data/sogou_news
tar -zvxf $DATAPATH/news_tensite_xml.smarty.tar.gz
cat $DATAPATH/news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  >  $DATAPATH/corpus.txt
python $DATAPATH/data_preprocess.py