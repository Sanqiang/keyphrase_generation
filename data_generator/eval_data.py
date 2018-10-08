from data_generator.vocab import Vocab
from util import constant
import json


class EvalData():
    def __init__(self, model_config):
        self.model_config = model_config
        if self.model_config.tied_embedding == 'enc|dec':
            self.voc_abstr = Vocab(self.model_config, vocab_path=self.model_config.path_abstrkword_voc)
            self.voc_kword = Vocab(self.model_config, vocab_path=self.model_config.path_abstrkword_voc)
        else:
            self.voc_abstr = Vocab(self.model_config, vocab_path=self.model_config.path_abstr_voc)
            self.voc_kword = Vocab(self.model_config, vocab_path=self.model_config.path_kword_voc)
        self.populate_data()
        self.size = len(self.data)
        if model_config.eval_mode == 'truncate2000':
            self.data = self.data[:2000]
            self.size = len(self.data)
            # assert self.size == 2000

    def populate_data(self):
        pad_id = self.voc_abstr.encode(constant.SYMBOL_PAD)
        self.data = []
        for line in open(self.model_config.path_val_json):
            obj = json.loads(line.strip())
            if self.model_config.subword_vocab_size > 0:
                abstr = self.voc_abstr.encode(' '.join(
                        [constant.SYMBOL_START] + obj['title'].split() + obj['abstr'].split() + [constant.SYMBOL_END]))
                if len(abstr) > self.model_config.max_abstr_len:
                    abstr = abstr[:self.model_config.max_abstr_len]
                else:
                    num_pad = self.model_config.max_abstr_len - len(abstr)
                    abstr.extend(num_pad * pad_id)
                kwords = [self.voc_kword.encode(' '.join(
                    [constant.SYMBOL_START] + kphrase.split() + [constant.SYMBOL_END]))
                    for kphrase in obj['kphrases'].split(';')]
                for kword_id, kword in enumerate(kwords):
                    if len(kword) >= self.model_config.max_kword_len:
                        kwords[kword_id] = kword[:self.model_config.max_kword_len]
                    else:
                        num_pad = self.model_config.max_kword_len - len(kword)
                        kwords[kword_id].extend(num_pad * pad_id)
            else:
                abstr = [self.voc_abstr.encode(w) for w
                         in [constant.SYMBOL_START] + obj['title'].split() + obj['abstr'].split() + [constant.SYMBOL_END]]
                if len(abstr) > self.model_config.max_abstr_len:
                    abstr = abstr[:self.model_config.max_abstr_len]
                else:
                    num_pad = self.model_config.max_abstr_len - len(abstr)
                    abstr.extend(num_pad * [pad_id])

                kwords = [[self.voc_kword.encode(w) for w
                           in [constant.SYMBOL_START] + kphrase.split() + [constant.SYMBOL_END]]
                          for kphrase in obj['kphrases'].split(';') ]
                for kword_id, kword in enumerate(kwords):
                    if len(kword) >= self.model_config.max_kword_len:
                        kwords[kword_id] = kword[:self.model_config.max_kword_len]
                    else:
                        num_pad = self.model_config.max_kword_len - len(kword)
                        kwords[kword_id].extend(num_pad * [pad_id])

            abstr_raw = [w for w
                         in [constant.SYMBOL_START] + obj['title'].split() + obj['abstr'].split() + [
                             constant.SYMBOL_END]]
            kwords_raw = [[w for w
                           in [constant.SYMBOL_START] + kphrase.split() + [constant.SYMBOL_END]]
                          for kphrase in obj['kphrases'].split(';')]

            obj = {
                'abstr': abstr,
                'abstr_raw': abstr_raw,
                'kwords': kwords,
                'kwords_raw': kwords_raw,
            }
            self.data.append(obj)

    def get_data_sample_it(self):
        i = 0
        while True:
            if i == self.size:
                yield None
            else:
                yield self.data[i]
                i += 1