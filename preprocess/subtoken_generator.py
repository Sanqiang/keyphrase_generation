from util.data.text_encoder import SubwordTextEncoder
from util.data import text_encoder

from collections import Counter

types = ['kword', 'abstr']
type = types[0]

dict = {}
for line in open('/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/%s.voc' % type):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    dict[word] = cnt

c = Counter(dict)
output_path = '/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/%s.subvoc' % type
sub_word = SubwordTextEncoder.build_to_target_size(50000, c, 1, 1e3,
                                                               num_iterations=100)
for i, subtoken_string in enumerate(sub_word._all_subtoken_strings):
    if subtoken_string in text_encoder.RESERVED_TOKENS_DICT:
        sub_word._all_subtoken_strings[i] = subtoken_string + '_'
sub_word.store_to_file(output_path)