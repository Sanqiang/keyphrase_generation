import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')
    parser.add_argument('-ngpus', '--num_gpus', default=1, type=int,
                        help='Number of GPU cards?')
    parser.add_argument('-bsize', '--batch_size', default=128, type=int,
                        help='Size of Mini-Batch?')
    parser.add_argument('-env', '--environment', default='crc',
                        help='The environment machine?')
    parser.add_argument('-out', '--output_folder', default='tmp',
                        help='Output folder?')
    parser.add_argument('-result', '--result_folder', default='result_val',
                        help='Result folder?')
    parser.add_argument('-warm', '--warm_start', default='',
                        help='Path for warm start checkpoint?')
    parser.add_argument('-upr', '--use_partial_restore', default=True, type=bool,
                        help='Whether to use partial restore?')

    parser.add_argument('-op', '--optimizer', default='adagrad',
                        help='Which optimizer to use?')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Value of learning rate?')
    parser.add_argument('-layer_drop', '--layer_prepostprocess_dropout', default=0.0, type=float,
                        help='Dropout rate for data input?')
    parser.add_argument('-cop', '--change_optimizer', default=False, type=bool,
                        help='Whether to change the optimizer?')
    parser.add_argument('-model_print_freq', '--model_print_freq', default=100, type=int,
                        help='Print frequency of model training?')


    # For Data
    parser.add_argument('-lc', '--lower_case', default=True, type=bool,
                        help='Whether to lowercase the vocabulary?')
    parser.add_argument('-mc', '--min_count', default=6, type=int,
                        help='Truncate the vocabulary less than equal to the count?')
    parser.add_argument('-svoc_size', '--subword_vocab_size', default=50000, type=int,
                        help='The size of subword vocabulary? if <= 0, not use subword unit.')
    parser.add_argument('-eval_freq', '--model_eval_freq', default=10000, type=int,
                        help='The frequency of evaluation at training? not use if = 0.')
    parser.add_argument('-itrain', '--it_train', default=False, type=bool,
                        help='Whether to iterate train data set?')
    parser.add_argument('-max_kword_len', '--max_kword_len', default=15, type=int,
                        help='Max of key word length?')
    parser.add_argument('-max_abstr_len', '--max_abstr_len', default=650, type=int,
                        help='Max of abstract length?')
    parser.add_argument('-max_cnt_kword', '--max_cnt_kword', default=10, type=int,
                        help='Max of key word count?')
    parser.add_argument('-emode', '--eval_mode', default='truncate2000',
                        help='Evaluation Mode?')
    parser.add_argument('-cmode', '--cov_mode', default='kp_attn',
                        help='Coverage Mode?')
    parser.add_argument('-temb', '--tied_embedding', default='',
                        help='Tied Embedding?')
    parser.add_argument('-pmode', '--pointer_mode', default='',
                        help='Pointer Mode?')

    # For Graph
    parser.add_argument('-dim', '--dimension', default=512, type=int,
                        help='Size of dimension?')
    parser.add_argument('-ns', '--number_samples', default=5000, type=int,
                        help='Number of samples used in Softmax?')
    parser.add_argument('-beam', '--beam_search_size', default=8, type=int,
                        help='Size of beam search?')

    # For Transformer
    parser.add_argument('-pos', '--hparams_pos', default='timing',
                        help='Whether to use positional encoding?')

    parser.add_argument('-nhl', '--num_hidden_layers', default=4, type=int,
                        help='Number of hidden layer?')
    parser.add_argument('-nel', '--num_encoder_layers', default=4, type=int,
                        help='Number of encoder layer?')
    parser.add_argument('-ndl', '--num_decoder_layers', default=4, type=int,
                        help='Number of decoder layer?')
    parser.add_argument('-nh', '--num_heads', default=4, type=int,
                        help='Number of multi-attention heads?')
    parser.add_argument('-penalty_alpha', '--penalty_alpha', default=0.6, type=float,
                        help='The alpha for length penalty?')

    # For Test
    parser.add_argument('-test_ckpt', '--test_ckpt', default='',
                        help='Path for test ckpt checkpoint?')
    parser.add_argument('-beam_search_strategy', '--beam_search_strategy', default='',
                        help='Beam search Mode?')


    args = parser.parse_args()
    return args


def list_config(config):
    attrs = [attr for attr in dir(config)
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    output = ''
    for attr in attrs:
        val = getattr(config, attr)
        output = '\n'.join([output, '%s=\t%s' % (attr, val)])
    return output


def get_path(file_path, env='crc'):
    if env == 'crc':
        return "/zfs1/hdaqing/saz31/keyphrase/tmp/" + file_path
    else:
        return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path


args = get_args()

class DummyConfig():
    warm_start = args.warm_start
    output_folder = args.output_folder
    subword_vocab_size = args.subword_vocab_size
    min_count = args.min_count
    num_gpus = args.num_gpus
    dimension = 128
    batch_size = 2
    pointer_mode = args.pointer_mode
    eval_mode = args.eval_mode
    cov_mode = args.cov_mode
    cov_mode = cov_mode.split(':')

    beam_search_size = args.beam_search_size
    number_samples = args.number_samples
    learning_rate = 0.01
    optimizer = args.optimizer
    max_grad_norm = 4.0
    num_heads = 2
    num_hidden_layers = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    hparams_pos = args.hparams_pos
    layer_prepostprocess_dropout = 0.0

    model_print_freq = args.model_print_freq

    logdir = get_path('../' + output_folder + '/log/', args.environment)
    modeldir = get_path('../' + output_folder + '/model/', args.environment)
    resultdir = get_path('../' + output_folder + '/result/', args.environment)

    path_train_json = get_path('data/dummy_train.json', 'sys')
    path_val_json = get_path('data/dummy_val.json', 'sys')

    tied_embedding = args.tied_embedding
    if subword_vocab_size > 0:
        path_abstr_voc = get_path('data/dummy_abstr.subvoc', 'sys')
        path_kword_voc = get_path('data/dummy_kword.subvoc', 'sys')
        max_kword_len = 15
        max_abstr_len = 200
        max_cnt_kword = 10
    else:
        if tied_embedding == 'enc|dec':
            path_abstrkword_voc = get_path('data/dummy_abstr.voc', 'sys')
        else:
            path_abstr_voc = get_path('data/dummy_abstr.voc', 'sys')
            path_kword_voc = get_path('data/dummy_kword.voc', 'sys')
        max_kword_len = 5
        max_abstr_len = 10
        max_cnt_kword = 10
    beam_search_strategy = args.beam_search_strategy


class DefaultConfig(DummyConfig):
    output_folder = args.output_folder
    logdir = get_path('../' + output_folder + '/log/', args.environment)
    modeldir = get_path('../' + output_folder + '/model/', args.environment)
    resultdir = get_path('../' + output_folder + '/result/', args.environment)
    learning_rate = args.learning_rate
    dimension = args.dimension
    batch_size = args.batch_size
    max_kword_len = args.max_kword_len
    max_abstr_len = args.max_abstr_len
    max_cnt_kword = args.max_cnt_kword
    num_heads = args.num_heads
    num_hidden_layers = args.num_hidden_layers
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    layer_prepostprocess_dropout = args.layer_prepostprocess_dropout
    # path_train_json = get_path(
    #     '../keyphrase_data/kp20k/ke20k_training.processed_rui.json', 'sys')
    # path_val_json = get_path(
    #     '../keyphrase_data/kp20k/ke20k_validation.processed_rui.json', 'sys')
    # path_test_json = get_path(
    #     '../keyphrase_data/kp20k/ke20k_testing.processed_rui.json', 'sys')
    path_train_json = get_path('../keyphrase_data/kp20k_cleaned/tf_data/kp20k.test.one2many.json', 'sys')
    path_val_json = get_path('../keyphrase_data/kp20k_cleaned/tf_data/kp20k.valid.one2many.json', 'sys')
    path_test_json = get_path('../keyphrase_data/kp20k_cleaned/tf_data/kp20k.test.one2many.json', 'sys')
    tied_embedding = args.tied_embedding
    # if tied_embedding == 'enc|dec':
    #     path_abstrkword_voc = get_path(
    #         '../keyphrase_data/kp20k/abstr_kword_rui.voc', 'sys')
    # else:
    # path_abstr_voc = get_path(
    #     '../keyphrase_data/kp20k/abstr.voc', 'sys')
    # path_kword_voc = get_path(
    #     '../keyphrase_data/kp20k/kword.voc', 'sys')
    path_abstr_voc = get_path('../keyphrase_data/kp20k_cleaned/tf_data/abstr.subvoc', 'sys')
    path_kword_voc = get_path('../keyphrase_data/kp20k_cleaned/tf_data/kword.subvoc', 'sys')


class DefaultValConfig(DefaultConfig):
    output_folder = args.output_folder
    result_folder = args.result_folder
    path_val_json = get_path('../keyphrase_data/kp20k_cleaned/tf_data/kp20k.valid.one2many.json', 'sys')
    # path_val_json = get_path(
    #     '../keyphrase_data/kp20k2/ke20k_validation.processed.json', 'sys')
    resultdir = get_path('../' + output_folder + '/' + result_folder + '/', args.environment)

class DefaultTestConfig(DefaultConfig):
    output_folder = args.output_folder
    result_folder = args.result_folder
    # path_val_json = get_path(
    #     '../keyphrase_data/kp20k2/ke20k_testing.processed.json', 'sys')
    path_test_json = get_path('../keyphrase_data/kp20k_cleaned/tf_data/kp20k.test.one2many.json', 'sys')
    resultdir = get_path('../' + output_folder + '/' + result_folder +'/', args.environment)

# class DefaultTestTruncated2000Config(DefaultConfig):
#     eval_mode = 'truncate2000'
#     beam_search_size = 200
#     output_folder = args.output_folder
#     path_val_json = get_path(
#         '../keyphrase_data/kp20k/ke20k_testing.processed.json', 'sys')
#     resultdir = get_path('../' + output_folder + '/result_test_truncate2000/', args.environment)
#
# class DefaultValTruncated2000Config(DefaultConfig):
#     eval_mode = 'truncate2000'
#     beam_search_size = 200
#     output_folder = args.output_folder
#     path_val_json = get_path(
#         '../keyphrase_data/kp20k/ke20k_validation.processed.json', 'sys')
#     resultdir = get_path('../' + output_folder + '/result_val_truncate2000/', args.environment)

