import argparse
import copy
import os


def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsuspported value!')

def argparse_console(argparser):
    """ Add argument by console
    Input:
        --argparser: ArgumentParser instance
    Output:
        --argparser: Argument-added Parser

    """
    # Stage and Directions
    argparser.add_argument('--build_data', type=str2bool,   default=True)
    argparser.add_argument('--train', type=str2bool,        default=True)
    argparser.add_argument('--test', type=str2bool,         default=False)
    argparser.add_argument('--train_dir',   default='../dataset/ud-treebanks-v2.1/ud-treebanks-v2.1', help='Traning dataset direction, language general.')
    argparser.add_argument('--test_dir',    default='../dataset/ud-treebanks-v2.1/ud-treebanks-v2.1', help='Testing dataset direction, language general.')
    argparser.add_argument('--embedd_dir',  default='../dataset/glove', help='Pre-trained embeddings direction.')
    argparser.add_argument('--elmo_dir',    default='../dataset/elmo', help='ELMo weights and options direction.')
    argparser.add_argument('--data_dir',    default='./data', help='Preprocessed data direction, including vocabulary-sub-dir and look-up-embedding-sub-dir. Vocabulary for word, char, POS, labels etc. direction.')
    argparser.add_argument('--result_dir',  default='./results', help='Results for params, model, prediction etc. direction.')
    argparser.add_argument('--device_ids',  default='0', help='Devices like 0-1-2')

    # Language and Task
    argparser.add_argument('--cross_domain', type=str2bool, default=False, help='Cross domain FLAG.')
    argparser.add_argument('--cross_lingual', type=str2bool,default=False, help='Cross lingual FLAG.')
    argparser.add_argument('--adversarial', type=str2bool,  default=False, help='Adversarial trained FLAG for corss-lingual or cross-domain.')
    argparser.add_argument('--multi_task', type=str2bool,   default=False, help='Multi-Task for Tagging.')
    argparser.add_argument('--data_model', choices=['seperate-data_seperate-model', 'seperate-data_combine-model', 'combine-data_combine-model'], default='seperate-data_seperate-model', help='Seperate or Combined Dataset or Model.')
    argparser.add_argument('-l', '--language', dest='lang_list', action='append',   help='Select abbreviated languages, e.g. -l en -l de -l zh , where \'-l\' is short for \'--language\' and support several options. ATTENTION: ONLY CROSS-LANGUAGE SUPPORT MULTI-LANGUAGE !!!')
    argparser.add_argument('-tb', '--treebank', dest='tree_list', action='append',  help='Select language-corresponding treebank where one language can have multi-domian treebanksi, and config-instance characterized by treebank.')
    argparser.add_argument('-t', '--task', dest='task', choices=['parse', 'pos', 'srl'], default='parse', help='Select Tasks in [parse, pos, srl], e.g.\" -t parse -t pos \".')

    # Embeddings
    argparser.add_argument('--enable_seqlabel', dest='seqlabFLAG', type=str2bool,          default=False)
    argparser.add_argument('--enable_word_ngram', dest='wngramFLAG', type=str2bool,        default=False)
    argparser.add_argument('--enable_word_pretrained', dest='wembeddFLAG', type=str2bool,  default=True)
    argparser.add_argument('--enable_elmo', dest='elmoFLAG', type=str2bool, default=False)
    argparser.add_argument('--enable_char', dest='charFLAG', type=str2bool, default=True)
    argparser.add_argument('--enable_upos', dest='uposFLAG', type=str2bool, default=True)
    argparser.add_argument('--enable_xpos', dest='xposFLAG', type=str2bool, default=True)
    argparser.add_argument('--reuse_subtoken', type=str2bool, default=False, help='Share char-embedd for cross-lingual')
    argparser.add_argument('--reuse_token', type=str2bool, default=False, help='Share word-embedd for mono-lang cross-domain')
    argparser.add_argument('--reduce_pretrained', type=str2bool, default=False, help='Reduce pretrained embeddings.')
    argparser.add_argument('--reduce_pretrained_dim', type=int, default=100, help='Reduce pretrained embeddings to dim.')
    argparser.add_argument('--reduce_elmo', type=str2bool, default=False, help='Reduce pretrained embeddings.')
    argparser.add_argument('--reduce_elmo_dim', type=int, default=100, help='Reduce pretrained embeddings to dim.')
    argparser.add_argument('--embedd_word_ngram', dest='dim_wngram', type=int, default=100, help='N-gram word embedding dimensions')
    argparser.add_argument('--embedd_word_pretrained', dest='dim_wembedd', type=int, default=100, help='Pretrained word dimensions')
    argparser.add_argument('--embedd_word_elmo', dest='dim_welmo', type=int, default=100, help='N-gram word embedding dimensions')
    argparser.add_argument('--embedd_char', dest='dim_char', type=int,          default=100, help='Charater embedding dimensions')
    argparser.add_argument('--embedd_postag', dest='dim_postag', type=int,      default=50, help='POS-Tags embedding dimensions')
    argparser.add_argument('--embedd_position', dest='dim_position', type=int,  default=100, help='Position embedding dimensions')
    argparser.add_argument('--embedd_dropout', choices=['global', 'local'],     default='local', help='Dropout over concat-embedds or separate embedds')
    argparser.add_argument('--embedd_keep_prob', type=float,    default=0.67, help='Dropout keep probability')
    argparser.add_argument('--char_nn', choices=['bilstm', 'cnn'], default='bilstm', help='Character Neural Network')
    argparser.add_argument('--char_rnn_nlayer', type=int,       default=1)
    argparser.add_argument('--char_rnn_cell_size', type=int,    default=50)
    argparser.add_argument('--char_keep_prob', type=float,      default=0.67)
    argparser.add_argument('--char_cnn_nlayer', type=int,       default=1)
    argparser.add_argument('--char_cnn_kernel_size', type=int,  default=3)
    argparser.add_argument('--char_cnn_nfilter', type=int,      default=100)
    argparser.add_argument('--elmo_loadtype', choices=['cache', 'intern'], default='intern', help='Load precomputed or interncomputed ELMo.')
    argparser.add_argument('--elmo_position', choices=['input', 'output', 'input-output'], default='input', help='Apply ELMo as input or output of task-specific Language Model.')

    # Encoder
    argparser.add_argument('--reuse_encoder', type=str2bool, default=False, help='Share char-embedd for cross-domain')
    argparser.add_argument('--encoder', choices=['birnn', 'transformer'], default='birnn', help='Encoder Type')
    # Bi-directional Reccurent Networks Encoder
    argparser.add_argument('--birnn_nlayer', type=int,  default=3)
    argparser.add_argument('--cell', choices=['lstm', 'sru', 'gru'], default='lstm', help='Recurrent Units Cell')
    argparser.add_argument('--cell_size', type=int,     default=300)
    argparser.add_argument('--varianceFLAG', type=str2bool,     default=False, help='Variational Dropout Wrapper for BiRNN')
    argparser.add_argument('--highwayFLAG', type=str2bool,      default=False, help='High-Way connection for BiRNN')
    argparser.add_argument('--local_attention', type=str2bool,  default=False, help='Self-Attention after each BiRNN-layer')
    argparser.add_argument('--global_attention', type=str2bool, default=False, help='Self-Attention after BiRNN-Encoder')
    argparser.add_argument('--input_keep_prob', type=float,  default=0.67, help='Dropout Wrapper input keep-prob')
    argparser.add_argument('--output_keep_prob', type=float, default=1.0, help='Dropout Wrapper output keep-prob')
    argparser.add_argument('--state_keep_prob', type=float,  default=0.67, help='Dropout Wrapper output keep-prob')
    argparser.add_argument('--attention_size', type=int,     default=100,  help='General Attention size for each attention layer')
    argparser.add_argument('--attention_type', choices=['additive', 'dot_product'], default='additive', help='Attention Type, if choose dot_product, please check norm-dot.')
    # Transformer (Self-Attention) Encoder
    argparser.add_argument('--enable_partition', type=str2bool, default=False, help='Devide input-embedd into content and position.')
    argparser.add_argument('--multi_head_combine', choices=['concat', 'weighted-add'], default='concat', help='Multi-head combined.')
    argparser.add_argument('--transformer_nlayer', type=int, default=2)
    argparser.add_argument('--nhead', type=int, default=8)
    argparser.add_argument('--model_size', type=int, default=512)
    argparser.add_argument('--pos_hidden_size', type=int, default=2048)
    argparser.add_argument('--residual_keep_prob', type=float, default=0.9)
    argparser.add_argument('--attention_keep_prob', type=float, default=1.0)
    argparser.add_argument('--relu_keep_prob', type=float, default=0.9)

    # Scoring and Decoding
    argparser.add_argument('--mlp_keep_prob', type=float, default=0.67, help='Dropout for All MLPs')
    argparser.add_argument('--reduce_encoder', type=str2bool, default=False, help='Dim-reduction FLAG after encoder')
    argparser.add_argument('--reduce_encoder_dim', type=int, default=200, help='Reduced-dim after encoder')
    argparser.add_argument('--arc_size', type=int, default=400, help='Size of edge classifier')
    argparser.add_argument('--rel_size', type=int, default=100, help='Size of label classifier')
    argparser.add_argument('--share_arc_rel', type=str2bool, default=False, help='Share head and modifier in arc and label, where arc and rel size must be the same.')
    argparser.add_argument('--share_arc_rel_size', type=int, default=400, help='Size of both edge and label when shared among head & modifier')
    argparser.add_argument('--score_attention', type=str2bool, default=False, help='Self-Attention over arc&rel score')
    argparser.add_argument('--decode', choices=['projective-eisner', 'non-projective-edmonds', 'non-projective-greedy'], default='non-projective-greedy', help='Decoding algorithm for projective-tree or non-projective-tree')
    argparser.add_argument('--objective', choices=['hinge_loss', 'cross_entropy'], default='cross_entropy', help='Objective as structual-loss or emperical-loss')

    # Training
    argparser.add_argument('--batch_by', choices=['token', 'sent'], default='sent', help='Batch formulation')
    argparser.add_argument('--batch_size', type=int, default=30, help='If batch by tokens, batch-size is 5000')
    argparser.add_argument('--nepoch', type=int, default=500, help='If batch by tokens, 5000 epoches, else 200')
    argparser.add_argument('--early_stop_epoches', type=int, default=30)
    argparser.add_argument('--optimizer', choices=['sgd', 'momentum', 'rms', 'adam', 'radam'], default='adam')
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--lr_decay', type=float, default=0.75)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.98)
    argparser.add_argument('--epsilon', type=float, default=1e-08)
    argparser.add_argument('--clip', type=float, default=5.0)
    argparser.add_argument('--l2_regular', type=str2bool, default=False, help='L2 Regularization')

    return argparser


def config_common(config):
    """ Common (language-general) Configure setting.
    Input:
        --config: argument dictionary

    """
    # direction checking
    if not os.path.exists(config.train_dir):
        raise ValueError('Training and validation direction doen not exist!')
    if not os.path.exists(config.test_dir):
        raise ValueError('Testing direction doen not exist!')
    if not os.path.exists(config.embedd_dir):
        raise ValueError('Pretrained Embedding direction doen not exist!')

    config.embedd_lookup_dir = config.data_dir + '/embedd_lookup'
    config.vocab_dir = config.data_dir + '/vocab'
    if not os.path.exists(config.embedd_lookup_dir):
        os.makedirs(config.embedd_lookup_dir)
    if not os.path.exists(config.vocab_dir):
        os.makedirs(config.vocab_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    devices = config.device_ids.split('-')
    config.devices = list(map(int, devices))

    config.languages = '-'.join(config.lang_list)
    config.vocab_char_shared = config.vocab_dir+'/'+config.languages+'.char.vocab'
    config.vocab_word_shared = config.vocab_dir+'/'+config.languages+'.word.vocab'
    config.vocab_emword_shared = config.vocab_dir+'/'+config.languages+'.emword.vocab'
    config.embedd_lookup_shared = config.embedd_lookup_dir+'/'+config.languages+'.embedd.lookup.npz'

    return config


def config_specific(gconfig, treebank='English', language='en'):
    """ Language Specific Configure Setting.
    Input:
        --gconfig: common configure
        --treebank: characterization of the specific config
        --language: conll-format language abbreviation
    Output:
        --config: language-specific configure
    """
    config = copy.deepcopy(gconfig)
    config.language = language
    config.treebank = treebank
    config.train_file = '%s/UD_%s/%s-ud-train.conllu'%(config.train_dir, config.treebank, config.language)
    config.dev_file   = '%s/UD_%s/%s-ud-dev.conllu'%(config.train_dir, config.treebank, config.language)
    config.test_file  = '%s/UD_%s/%s-ud-test.conllu'%(config.test_dir, config.treebank, config.language)
    #config.embedd = '%s/wiki.%s.vec'%(config.embedd_dir, config.language)
    config.embedd = '%s/glove.6B.%dd.txt'%(config.embedd_dir, config.dim_wembedd)

    if not os.path.exists(config.train_file):
        raise ValueError('Training data for %s doen not exist!'%(config.treebank))
    if not os.path.exists(config.dev_file):
        raise ValueError('Validation data for %s doen not exist!'%(config.treebank))
    if not os.path.exists(config.test_file):
        raise ValueError('Testing data for %s doen not exist!'%(config.treebank))
    if not os.path.exists(config.embedd):
        raise ValueError('Pretrained Embedding for %s doen not exist!'%(config.language))

    config.vocab_dir = config.vocab_dir+'/'+config.treebank
    config.embedd_lookup_dir = config.embedd_lookup_dir+'/'+config.treebank
    config.result_dir = config.result_dir+'/'+config.treebank
    if not os.path.exists(config.vocab_dir):
        os.makedirs(config.vocab_dir)
    if not os.path.exists(config.embedd_lookup_dir):
        os.makedirs(config.embedd_lookup_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    config.vocab_word = config.vocab_dir+'/word.vocab'
    config.vocab_char = config.vocab_dir+'/char.vocab'
    config.vocab_upos = config.vocab_dir+'/upos.vocab'
    config.vocab_xpos = config.vocab_dir+'/xpos.vocab'
    config.vocab_lemma  = config.vocab_dir+'/lemma.vocab'
    config.vocab_deprel = config.vocab_dir+'/deprel.vocab'
    config.vocab_emword  = config.vocab_dir+'/emword.vocab'
    config.vocab_seqlab  = config.vocab_dir+'/seqlab.vocab'

    config.embedd_lookup = config.embedd_lookup_dir+'/embedd.lookup.npz'
    config.elmo_options  = config.elmo_dir+'/options.json'
    config.elmo_weights  = config.elmo_dir+'/weights.hdf5'

    config.log = config.result_dir+'/model.log'
    config.checkpoint = config.result_dir+'/model.ckp'
    config.model_saved = config.result_dir+'/model.weights'
    config.pred_results = config.result_dir+'/preds.conllu'
    config.target_results = config.result_dir+'/target.conllu'

    return config
