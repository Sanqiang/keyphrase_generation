# For fix slurm cannot load PYTHONPATH
import sys

sys.path.insert(0, '/zfs1/hdaqing/saz31/kp/keyphrase')


import numpy as np
import glob
from data_generator.eval_data import EvalData
from model.graph import Graph
import tensorflow as tf
from util.checkpoint import copy_ckpt_to_modeldir
from util.decode import decode_keyphrases, decode_gt_keyphrase
from util.f1 import calculate_f1
import time
from os import mkdir
import tensorflow.contrib.slim as slim
from util import constant
from os.path import exists, join
from os import makedirs, listdir, remove

def get_feed(objs, it, model_config, data):
    input_feed = {}
    assert len(objs) == 1
    exclude_size = 0
    obj = objs[0]
    tmp_abstr, tmp_kword = [], []
    tmp_abstr_raw, tmp_kword_raw = [], []
    is_finished = False

    for i in range(model_config.batch_size):
        data_sample = next(it)
        if data_sample is None:
            pad_id = data.voc_abstr.encode(constant.SYMBOL_PAD)
            data_sample = {
                'abstr': model_config.max_abstr_len * [pad_id],
                'abstr_raw': model_config.max_abstr_len * [constant.SYMBOL_PAD],
                'kwords': [model_config.max_kword_len * [pad_id] for _ in
                           range(model_config.max_cnt_kword)],
                'kwords_raw': [model_config.max_kword_len * [constant.SYMBOL_PAD] for _ in
                               range(model_config.max_cnt_kword)],
                'oov': {'w2i': {}, 'i2w': []}
            }
            is_finished = True
            exclude_size += 1
        assert len(data_sample['abstr']) == model_config.max_abstr_len

        tmp_abstr.append(data_sample['abstr'])
        tmp_abstr_raw.append(data_sample['abstr_raw'])
        kwords_tmp = data_sample['kwords']
        kwords_tmp_raw = data_sample['kwords_raw']
        while len(kwords_tmp) < model_config.max_cnt_kword:
            kwords_tmp.append([1] * len(kwords_tmp[0]))
            kwords_tmp_raw.append([constant.SYMBOL_PAD] * len(kwords_tmp[0]))
        tmp_kword.append(kwords_tmp)
        tmp_kword_raw.append(kwords_tmp_raw)

    for step in range(model_config.max_abstr_len):
        input_feed[obj['abstr_ph'][step].name] = [
            tmp_abstr[batch_idx][step] for batch_idx in range(model_config.batch_size)]

    for kword_idx in range(model_config.max_cnt_kword):
        for step in range(model_config.max_kword_len):
            input_feed[obj['kwords_ph'][kword_idx][step].name] = [
                tmp_kword[batch_idx][kword_idx][step] for batch_idx in range(model_config.batch_size)]

    return input_feed, tmp_kword_raw, is_finished, exclude_size

# Post Process for eval (stemmer)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def kphrases_process(kp_list):
    return [_kp_eval_process(kp) for kp in kp_list]


def _kp_eval_process(kp):
    return ' '.join([stemmer.stem(w.strip().lower()) for w in kp.split()])
# Post Process for eval (stemmer)


def eval(model_config, ckpt):
    evaldata = EvalData(model_config)
    it = evaldata.get_data_sample_it()
    tf.reset_default_graph()
    graph = Graph(model_config, False, evaldata)
    graph.create_model_multigpu()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    restore_op = slim.assign_from_checkpoint_fn(
        ckpt, slim.get_variables_to_restore(),
        ignore_missing_vars=False, reshape_variables=False)
    def init_fn(session):
        restore_op(session)
        # graph.saver.restore(session, ckpt)
        print('Restore ckpt:%s.' % ckpt)

    sv = tf.train.Supervisor(init_fn=init_fn)
    sess = sv.PrepareSession(config=config)
    perplexitys = []
    prec_top10s, recall_top10s, f1_top10s = [], [], []
    prec_top5s, recall_top5s, f1_top5s = [], [], []
    prec_tunes, recall_tunes, f1_tunes = [], [], []
    reports = []
    while True:
        input_feed, gt_kphrases, is_finished, exclude_size = get_feed(graph.objs, it, model_config, evaldata)
        if is_finished:
            break
        # s_time = datetime.now()
        fetches = [graph.loss, graph.global_step, graph.perplexity,
                   graph.objs[0]['targets'], graph.objs[0]['pred_occupies']]
        loss, step, perplexity, targets, pred_occupies = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        for batch_i in range(model_config.batch_size - exclude_size):
            target = targets[batch_i]
            kphrases_topn = decode_keyphrases(target, evaldata, model_config, topk=model_config.max_cnt_kword)
            kphrases_top10 = kphrases_topn[:10]
            kphrases_gt = decode_gt_keyphrase(gt_kphrases[batch_i], evaldata, model_config)
            kphrases_top5 = kphrases_top10[:5] if len(kphrases_top10) > 5 else kphrases_top10

            kphrases_top10_stem = kphrases_process(kphrases_top10)
            kphrases_top5_stem = kphrases_process(kphrases_top5)
            kphrases_gt_stem = kphrases_process(kphrases_gt)
            prec_top10, recall_top10, f1_top10 = calculate_f1(set(kphrases_top10_stem), set(kphrases_gt_stem))
            prec_top5, recall_top5, f1_top5 = calculate_f1(set(kphrases_top5_stem), set(kphrases_gt_stem))

            prec_top10s.append(prec_top10)
            recall_top10s.append(recall_top10)
            f1_top10s.append(f1_top10)

            prec_top5s.append(prec_top5)
            recall_top5s.append(recall_top5)
            f1_top5s.append(f1_top5)

            kphrases_tune = []
            for pred_idx, pred_occupy in enumerate(pred_occupies[batch_i]):
                if pred_occupy >= 0.5:
                    if pred_idx < len(kphrases_topn):
                        kphrases_tune.append(kphrases_topn[pred_idx])
                else:
                    break
            kphrases_tune_stem = kphrases_process(kphrases_tune)
            prec_tune, recall_tune, f1_tune = calculate_f1(
                set(kphrases_tune_stem), set(kphrases_gt_stem))
            prec_tunes.append(prec_tune)
            recall_tunes.append(recall_tune)
            f1_tunes.append(f1_tune)

            report = 'pred@10:%s\npred@tune:%s\ngt:%s\n\n\n' %\
                     (';'.join(kphrases_top10), ';'.join(kphrases_tune), ';'.join(kphrases_gt))
            reports.append(report)

        # e_time = datetime.now()
        # span = e_time - s_time
        # print('%s' % (str(span)))
    format = '%.4f'
    file_name = ''.join(['step_', str(step),
                         'f1tune_', str(format % np.mean(f1_tunes)),'prectune_', str(format % np.mean(prec_tunes)),
                         'recalltune_', str(format % np.mean(recall_tunes)),
                         'f1top10_', str(format % np.mean(f1_top10s)), 'f1top5_', str(format % np.mean(f1_top5s)),
                         'prectop10_', str(format % np.mean(prec_top10s)), 'prectop5_', str(format % np.mean(prec_top5s)),
                         'recalltop10_', str(format % np.mean(recall_top10s)), 'recalltop5_', str(format % np.mean(recall_top5s)),
                         'perplexity_', str(np.mean(perplexitys))
                         ])
    if not exists(model_config.resultdir):
        mkdir(model_config.resultdir)
    f = open(model_config.resultdir + file_name, 'w')
    f.write('\n\n'.join(reports))
    f.close()

    return np.mean(f1_tunes)


def get_ckpt(modeldir, logdir, wait_second=60):
    while True:
        try:
            ckpt = copy_ckpt_to_modeldir(modeldir, logdir)
            return ckpt
        except FileNotFoundError as exp:
            if wait_second:
                print(str(exp) + '\nWait for 1 minutes.')
                time.sleep(wait_second)
            else:
                return None


def get_best_f1(model_config):
    if not exists(model_config.resultdir):
        makedirs(model_config.resultdir)
    best_acc_file = join(model_config.resultdir, 'best_f1')
    if exists(best_acc_file):
        return float(open(best_acc_file).readline())
    else:
        return 0.0


def write_best_f1(model_config, acc):
    best_acc_file = join(model_config.resultdir, 'best_f1')
    open(best_acc_file, 'w').write(str(acc))


if __name__ == '__main__':
    from model.model_config import DefaultConfig, DefaultValConfig, DefaultTestConfig, DummyConfig
    model_config = DefaultConfig()
    best_f1 = get_best_f1(model_config)
    while True:
        ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
        if ckpt:
            if model_config.eval_mode == 'none':
                # f1 = eval(DummyConfig(), ckpt)
                f1 = eval(DefaultValConfig(), ckpt)
            elif model_config.eval_mode == 'truncate2000':
                # f1 = eval(DummyConfig(), ckpt)
                f1 = eval(DefaultValConfig(), ckpt)
                # eval(DefaultTestConfig(), ckpt)

            if f1 > best_f1:
                best_f1 = f1
                write_best_f1(model_config, best_f1)
                for file in listdir(model_config.modeldir):
                    step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                    if step not in file:
                        remove(model_config.modeldir + file)
            else:
                for fl in glob.glob(ckpt + '*'):
                    remove(fl)
