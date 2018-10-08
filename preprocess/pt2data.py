"""Transfer torch pt to raw data"""
import torch
import json
from datetime import datetime
from collections import Counter

tasks = ['train', 'valid', 'test']

for task in tasks:
    PATH = '/Users/sanqiangzhao/git/keyphrase_data/kp20k_cleaned/pt_data/kp20k/kp20k.%s.one2many.pt' % task
    VOCAB_PATH = '/Users/sanqiangzhao/git/keyphrase_data/kp20k_cleaned/pt_data/kp20k/kp20k.vocab.pt'
    NPATH = '/Users/sanqiangzhao/git/keyphrase_data/kp20k_cleaned/tf_data/kp20k.%s.one2many.json' % task

    # Load vocab
    word2id, id2word, vocab = torch.load(VOCAB_PATH, 'rb')
    is_train = 'kp20k.train.one2many.pt' in PATH

    data = torch.load(PATH, 'rb')

    nlines = []
    s_time = datetime.now()
    c_src, c_trg = Counter(), Counter()

    for obj_id, obj in enumerate(data):
        src = obj['src_str']
        trg = obj['trg_str']
        src_text = ' '.join(src)
        trg_text = ';'.join([' '.join(kp) for kp in trg])

        c_src.update(src)
        c_trg.update([item for sublist in trg for item in sublist])

        nobj = {}
        nobj['title'] = ''
        nobj['abstr'] = src_text
        nobj['kphrases'] = trg_text
        nline = json.dumps(nobj)
        nlines.append(nline)

        if obj_id % 1000 == 0:
            e_time = datetime.now()
            span = e_time - s_time
            s_time = datetime.now()
            print('Process %s using %s' % (obj_id, span))

    open(NPATH, 'w').write('\n'.join(nlines))

    if is_train:
        # Generate vocab
        v_lines = []
        for w, cnt in c_src.most_common():
            v_lines.append('%s\t%s' % (w, str(cnt)))
        open('/Users/sanqiangzhao/git/keyphrase_data/kp20k/src.vocab', 'w').write('\n'.join(nlines))

        v_lines = []
        for w, cnt in c_trg.most_common():
            v_lines.append('%s\t%s' % (w, str(cnt)))
        open('/Users/sanqiangzhao/git/keyphrase_data/kp20k/trg.vocab', 'w').write('\n'.join(nlines))




