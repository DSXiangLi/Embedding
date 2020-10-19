#!/usr/bin/env bash
DIR='bookcorpus'
DATADIR='data/bookcorpus'

python -u $DIR/download_list.py > $DATADIR/url_list.jsonl &
python $DIR/download_files.py --list $DATADIR/url_list.jsonl --out $DATADIR/out_txts
python $DIR/make_sentlines.py $DATADIR/out_txts | python $DIR/tokenize_sentlines.py > $DATADIR/all.tokenized.txt