## SQUAD
 
Website: https://rajpurkar.github.io/SQuAD-explorer/

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

数据处理
1. 考虑到transformer主要是后面用在USE作为特征提取，这里暂不考虑impossible to answer的问题
2. Dev数据每个问题包含多个答案，这里对重复回答只保留一个 
