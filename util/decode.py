from util import constant


def decode_keyphrases(target, evaldata, model_config, topk=10, oov=None):
    """Get a list of predicted keyphrases"""
    keyphrases = []
    bos_id = evaldata.voc_kword.encode('#bos#')
    eos_id = evaldata.voc_kword.encode('#eos#')
    # assert topk < len(target)
    for i in range(len(target)):
        keyphrase = decode_keyphrase(target[i], evaldata, model_config, bos_id, eos_id, oov=oov)
        if keyphrase not in keyphrases and constant.SYMBOL_UNK not in keyphrase:
            keyphrases.append(keyphrase)
        if len(keyphrases) >= topk:
            break

    return keyphrases


def decode_gt_keyphrase(gt_kphrases, evaldata, model_config):
    keyphrases = []
    bos_id = constant.SYMBOL_START
    eos_id = constant.SYMBOL_END
    for i in range(len(gt_kphrases)):
        keyphrase = gt_kphrases[i]
        # keyphrase = decode_keyphrase(gt_kphrases[i], evaldata, model_config, bos_id, eos_id)
        if bos_id in keyphrase:
            left_idx = keyphrase.index(bos_id)+1
        else:
            left_idx = 0
        if eos_id in keyphrase:
            right_idx = keyphrase.index(eos_id)
        else:
            right_idx = len(keyphrase) - 1
        keyphrase = ' '.join([w for w in keyphrase[left_idx:right_idx] if w != constant.SYMBOL_PAD])
        if keyphrase:
            keyphrases.append(keyphrase)
    return keyphrases


def decode_keyphrase(keyphrase, evaldata, model_config, bos_id=3, eos_id=4, oov=None):
    """Get clean key phrase"""
    if model_config.subword_vocab_size > 0:
        keyphrase = list(keyphrase)
        if bos_id in keyphrase:
            left_idx = keyphrase.index(bos_id)+1
        else:
            left_idx = 0
        if eos_id in keyphrase:
            right_idx = keyphrase.index(eos_id)
        else:
            right_idx = len(keyphrase) - 1
        keyphrase = keyphrase[left_idx:right_idx]
        return evaldata.voc_kword.describe(keyphrase)
    else:
        keyphrase = list(keyphrase)
        if bos_id in keyphrase:
            left_idx = keyphrase.index(bos_id)+1
        else:
            left_idx = 0
        if eos_id in keyphrase:
            right_idx = keyphrase.index(eos_id)
        else:
            right_idx = len(keyphrase) - 1
        keyphrase = keyphrase[left_idx:right_idx]
        return ' '.join([evaldata.voc_kword.describe(wd, oov=oov) for wd in keyphrase if evaldata.voc_kword.describe(wd) != constant.SYMBOL_PAD])
