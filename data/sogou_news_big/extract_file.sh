#!/usr/bin/env bash
tar -zvxf news_sohusite_xml.full.tar.gz
cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  > corpus.txt