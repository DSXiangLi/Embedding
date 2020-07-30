#!/usr/bin/env bash
tar -zvxf news_tensite_xml.smarty.tar.gz
cat news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  > corpus.txt