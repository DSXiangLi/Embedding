# -*-coding:utf-8 -*-
import os
import jieba
from pathos.multiprocessing import ProcessingPool
from data.preprocess_util import str_utils_en, str_utils_ch, dump_dictionary

def merge_blanks(src, targ, verbose=False):
    """
    text merge credit to https://github.com/twairball/t2t_wmt_zhen
    without merge, English and Chinese lines count mismatch
    """
    merges_done = []  # array of indices of rows merged
    sub = None  # replace sentence after merge

    is_sgm = '.sgm' in src
    with open(src, 'rb') as src_file, open(targ, 'rb') as targ_file:
        src_lines = src_file.readlines()
        targ_lines = targ_file.readlines()

        print("src: %d, targ: %d" % (len(src_lines), len(targ_lines)))
        print("=" * 30)
        for i in range(0, len(src_lines) - 1):
            s = _preprocess_sgm(src_lines[i].decode('utf-8').rstrip(), is_sgm)
            s_next = _preprocess_sgm(src_lines[i + 1].decode('utf-8').rstrip(), is_sgm)

            t = _preprocess_sgm(targ_lines[i].decode('utf-8').rstrip(), is_sgm)
            t_next = _preprocess_sgm(targ_lines[i + 1].decode('utf-8').rstrip(), is_sgm)
            if t == '.':
                t = ''
            if t_next == '.':
                t_next = ''

            if (len(s_next) == 0) and (len(t_next) > 0):
                targ_lines[i] = "%s %s" % (t, t_next)  # assume it has punctuation
                targ_lines[i + 1] = b''
                src_lines[i] = s if len(s) > 0 else sub

                merges_done.append(i)
                if verbose:
                    print("t [%d] src: %s\n      targ: %s" % (i, src_lines[i], targ_lines[i]))
                    print()

            elif (len(s_next) > 0) and (len(t_next) == 0):
                src_lines[i] = "%s %s" % (s, s_next)  # assume it has punctuation
                src_lines[i + 1] = b''
                targ_lines[i] = t if len(t) > 0 else sub

                merges_done.append(i)
                if verbose:
                    print("s [%d] src: %s\n      targ: %s" % (i, src_lines[i], targ_lines[i]))
                    print()
            elif (len(s) == 0) and (len(t) == 0):
                # both blank -- remove
                merges_done.append(i)
            else:
                src_lines[i] = s if len(s) > 0 else sub
                targ_lines[i] = t if len(t) > 0 else sub

        # handle last line
        s_last = src_lines[-1].decode('utf-8').strip()
        t_last = targ_lines[-1].decode('utf-8').strip()
        if (len(s_last) == 0) and (len(t_last) == 0):
            merges_done.append(len(src_lines) - 1)
        else:
            src_lines[-1] = s_last
            targ_lines[-1] = t_last

    # remove empty sentences
    for m in reversed(merges_done):
        del src_lines[m]
        del targ_lines[m]

    print("merges done: %d" % len(merges_done))
    return (src_lines, targ_lines)


def _preprocess_sgm(line, is_sgm):
  """Preprocessing to strip tags in SGM files."""
  if not is_sgm:
    return line
  # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
  if line.startswith("<srcset") or line.startswith("</srcset"):
    return ""
  if line.startswith("<doc") or line.startswith("</doc"):
    return ""
  if line.startswith("<p>") or line.startswith("</p>"):
    return ""
  # Strip <seg> tags.
  line = line.strip()
  if line.startswith("<seg") and line.endswith("</seg>"):
    i = line.index(">")
    return line[i + 1:-6]  # Strip first <seg ...> and last </seg>.
  return ""


def filter_sample(source_sentences, target_sentences):
    """
    To make model training easier, here sample trimed is done, can be alter to other usage
    """
    new_source = []
    new_target = []
    for source, target in zip(source_sentences, target_sentences):
        if (len(source)<=5) or (len(target)<=5):
            continue
        elif (len(source)>20) or(len(target)>20):
            continue
        else:
            new_source.append(source)
            new_target.append(target)
    return new_source, new_target


def main(data_dir, file_dict, surfix, dry_run_dict):
    encoder_path = '{}/{}_encoder_source.txt'.format(data_dir, surfix)
    decoder_path = '{}/{}_decoder_source.txt'.format(data_dir, surfix)

    source_sentences, target_sentences = merge_blanks(os.path.join(data_dir, file_dict['source']),
                                                      os.path.join(data_dir, file_dict['target']))

    print('String Preprocessing')
    source_sentences = str_utils_en.text_cleaning(source_sentences)
    target_sentences = str_utils_ch.text_cleaning(target_sentences)
    print('Double check source={}, target={}'.format(len(source_sentences), len(target_sentences)))

    print('Word segmentation')
    jieba.initialize()
    jieba.disable_parallel()
    with ProcessingPool(nodes=min(os.cpu_count(), 5)) as pool:
        source_sentences = pool.map(lambda x: [i.strip() for i in x.strip().lower().split(' ') if len(i)>=1], source_sentences)
    with ProcessingPool(nodes=min(os.cpu_count(), 5)) as pool:
        target_sentences = pool.map(lambda x: [i.strip() for i in jieba.cut(x.strip(), cut_all=False) if len(i)>=1], target_sentences)
    print('Triple check source={}, target={}'.format(len(source_sentences), len(target_sentences)))

    source_sentences, target_sentences = filter_sample(source_sentences, target_sentences)
    print('Triple check source={}, target={}'.format(len(source_sentences), len(target_sentences)))
    print('Writing pair into encoder and decoder source at {}'.format(data_dir))
    with open(encoder_path, 'w', encoding='utf-8') as fe, open(decoder_path, 'w', encoding='utf-8') as fd :
        for encoder_source, decoder_source in zip(source_sentences, target_sentences):
            fe.write(' '.join(encoder_source).lower())
            fe.write('\n')
            fd.write(' '.join(decoder_source).lower())
            fd.write('\n')

    # better sub tokenizer can be used to generate dictionary
    dump_dictionary(data_dir, source_sentences, prefix='source', debug=True, dry_run=dry_run_dict)
    dump_dictionary(data_dir, target_sentences, prefix='target', debug=True, dry_run=dry_run_dict)


if __name__=='__main__':
    data_dir = './data/wmt'
    const_dir = 'const'

    file_dict = {'source': 'news-commentary-v12.zh-en.en',
                 'target': 'news-commentary-v12.zh-en.zh'}
    main(data_dir, file_dict, surfix='train', dry_run_dict=False)

    file_dict = {'source': 'newsdev2017-enzh-src.en.sgm',
                 'target': 'newsdev2017-enzh-ref.zh.sgm'}
    main(data_dir, file_dict,  surfix='dev', dry_run_dict=True)
