"""Valid proper length"""
import json
from collections import Counter
from data_generator.vocab import Vocab
from model.model_config import DefaultConfig

PATH = '/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/kp20k.valid.one2many.json'

voc_abstr = Vocab(DefaultConfig(), vocab_path='/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/abstr.subvoc')
voc_kword = Vocab(DefaultConfig(), vocab_path='/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/kword.subvoc')

max_abstr_len, max_kword_len, max_kword_cnt = 0, 0, 0
c_abstr_len, c_kwod_len, c_kword_cnt = Counter(), Counter(), Counter()

for line in open(PATH):
    obj = json.loads(line)
    kphrases = obj['kphrases'].split(';')
    abstr = obj['abstr']

    abstr_ids = voc_abstr.encode(abstr)
    if len(abstr_ids) > max_abstr_len:
        print(abstr)
    max_abstr_len = max(max_abstr_len, len(abstr_ids))
    c_abstr_len.update([len(abstr_ids)])

    max_kword_cnt = max(max_kword_cnt, len(kphrases))
    c_kword_cnt.update([len(kphrases)])

    for kphrase in kphrases:
        max_kword_len = max(max_kword_len, len(voc_kword.encode(kphrase)))
        c_kwod_len.update([len(voc_kword.encode(kphrase))])

print(max_abstr_len)
print(max_kword_len)
print(max_kword_cnt)

print(c_abstr_len.most_common())
print(c_kwod_len.most_common())
print(c_kword_cnt.most_common())