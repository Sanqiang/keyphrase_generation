import json
from collections import Counter
import spacy

f_abstr_voc = open('/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/abstr.voc', 'w')
f_kword_voc = open('/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/kword.voc', 'w')

c_abstr = Counter()
c_kword = Counter()

for line in open('/Users/sanqiangzhao/git/kp/keyphrase_data/kp20k_cleaned/tf_data/kp20k.train.one2many.json'):
    obj = json.loads(line.lower())
    words_abstr = [w for w in obj['title'].split()] + [w for w in obj['abstr'].split()]
    c_abstr.update(words_abstr)
    for kphrase in obj['kphrases'].split(';'):
        words_kphrase = [w for w in kphrase.split()]
        c_kword.update(words_kphrase)


for word, cnt in c_abstr.most_common():
    f_abstr_voc.write(word)
    f_abstr_voc.write('\t')
    f_abstr_voc.write(str(cnt))
    f_abstr_voc.write('\n')
f_abstr_voc.close()


for word, cnt in c_kword.most_common():
    f_kword_voc.write(word)
    f_kword_voc.write('\t')
    f_kword_voc.write(str(cnt))
    f_kword_voc.write('\n')
f_kword_voc.close()