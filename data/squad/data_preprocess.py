# -*- coding=utf-8 -*-
import os
from data.preprocess_util import *
import json
import pickle


def json2record(data_dir, surfix):
    file = json.load(open('{}/{}-v2.0.json'.format(data_dir, surfix)))
    data = file['data']

    result = []
    for paragraph in data:
        title = paragraph['title']
        for item in paragraph['paragraphs']:
            context = item['context']
            for qa in item['qas']:
                question = qa['question']
                id = qa['id']
                is_impossible = qa['is_impossible']
                for answers in qa['answers']:
                    answer = answers['text']
                    result.append({
                        'id': id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answer': answer,
                        'is_impossible': is_impossible
                    })
    with open('{}/{}.pkl'.format(data_dir, surfix), 'wb') as f:
        pickle.dump(result, f)


def main(data_dir, const_dir, language, surfix):
    with open('{}/{}.pkl'.format(data_dir, surfix), 'rb') as f:
        record = pickle.load(f)

    encoder_path = '{}/{}_encoder_source.txt'.format(data_dir, surfix)
    decoder_path = '{}/{}_decoder_source.txt'.format(data_dir, surfix)

    preprocess = StrUtils(os.path.join(const_dir, language), language)

    print('String Preprocessing and word Segmentation')
    question = preprocess.text_cleaning([i['question'] for i in record if not i['is_impossible']])
    question = preprocess.multi_word_cut(question)

    answer = preprocess.text_cleaning([i['answer'] for i in record if not i['is_impossible']])
    answer = preprocess.multi_word_cut(answer)

    qa = zip(question, answer)
    qa = sorted(qa, key= lambda x: len(x[0])) # sort train sample using question len

    print('Writing QA into encoder and decoder source at {}'.format(data_dir))
    dup = {}

    with open(encoder_path, 'w', encoding='utf-8') as fe, open(decoder_path, 'w', encoding='utf-8') as fd:
        for encoder_source, decoder_source in qa:
            if len(encoder_source)>0 and len(decoder_source) >0:
                enc = ' '.join(encoder_source).lower()
                dec = ' '.join(decoder_source).lower()
                if enc in dup:
                    if dec in dup[enc]:
                        continue
                    else:
                        dup[enc].append(dec)
                else:
                    dup[enc] = [dec]
                fe.write(enc)
                fe.write('\n')
                fd.write(dec)
                fd.write('\n')

    print('Dumping Original Dictionary')
    dump_dictionary(data_dir, question+answer, debug=True, dry_run=(surfix != 'train'))# only dump dictionry for train


if __name__=='__main__':
    data_dir = './data/squad'
    language = 'en'
    const_dir = 'const'
    json2record(data_dir, 'train')
    json2record(data_dir, 'dev')
    main(data_dir, const_dir, language, 'train')
    main(data_dir, const_dir, language, 'dev')
