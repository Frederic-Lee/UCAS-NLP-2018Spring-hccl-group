import argparse
import copy
import os

from config import *
from utils.dict2obj import Dict2Obj
from data_process.vocab import *
from data_process.dataset import ConllDataset
from data_process.collector import DataCollector
from models.base_model import BaseModel
from models.parser import Parser
from models.tagger import Tagger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2-3-4"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == '__main__':
    """ Argument Parser Settings """
    # argparser from console
    argparser = argparse.ArgumentParser(prog="Universal Dependencies")
    argparser = argparse_console(argparser)

    # general config object and saving to file
    args = argparser.parse_args()
    args_dict = args.__dict__
    fconfig = './arguments.cfg'
    gconfig = Dict2Obj(args_dict, recur=False)
    gconfig = config_common(gconfig)
    with open(fconfig, 'w') as fcfg:
        for key, value in args_dict.items():
            argline = '%s \t = %s'%(key, value)
            fcfg.write(argline+'\n')
            print (argline)

    # treebank-specific config object
    config_list = []
    for treebank, language in zip(gconfig.tree_list, gconfig.lang_list):
        print ("Constructing config for %s-%s..."%(language, treebank))
        config = config_specific(gconfig, treebank, language)
        print(config.vocab_dir, gconfig.vocab_dir)
        config_list.append(config)


    """ Dataset Preprocessing """
    if gconfig.build_data:
        for config in config_list:
            # Generate datasets, pay attention to lowercase and digit
            train = ConllDataset(config.train_file, convert2idx=False, enable_seqlabel=gconfig.seqlabFLAG)
            dev   = ConllDataset(config.dev_file, convert2idx=False, enable_seqlabel=gconfig.seqlabFLAG)
            test  = ConllDataset(config.test_file, convert2idx=False, enable_seqlabel=gconfig.seqlabFLAG)
            # Build token-level vocabs
            vocab_word, vocab_char, vocab_lemma, vocab_upos, vocab_xpos, vocab_deprel, vocab_seqlab = get_token_vocab([train, dev, test], seqlabFLAG=gconfig.seqlabFLAG)
            vocab_embedd = get_embedd_vocab(config.embedd)
            vocab_emword = vocab_word & vocab_embedd
            # Save vocab
            save_vocab(vocab_word, config.vocab_word)
            save_vocab(vocab_char, config.vocab_char)
            save_vocab(vocab_upos, config.vocab_upos)
            save_vocab(vocab_xpos, config.vocab_xpos)
            save_vocab(vocab_lemma, config.vocab_lemma)
            save_vocab(vocab_deprel, config.vocab_deprel)
            save_vocab(vocab_emword, config.vocab_emword)
            if gconfig.seqlabFLAG and len(vocab_seqlab)>0:
                save_vocab(vocab_seqlab, config.vocab_seqlab)

            if not gconfig.reuse_token:
                # Encode word vocab to pre-trained embeddings
                vocab_emword = load_vocab(config.vocab_emword)
                save_embedd_lookup(vocab_emword, config.embedd, config.embedd_lookup, config.dim_wembedd)

        # Union char-vocabs, support cross-lingual and cross-domain
        if gconfig.reuse_subtoken:
            try:
                union_vocab('char', config_list, gconfig.vocab_char_shared)
            except IOError:
                print ('Make sure enabled reuse_subtoken, and shared-vocab is %s'%(gconfig.vocab_char_shared))
        # Union word & emword vocabs and embeddings, not support cross-lingual yet
        if gconfig.reuse_token:
            try:
                union_vocab('word', config_list, gconfig.vocab_word_shared)
                union_vocab('emword', config_list, gconfig.vocab_emword_shared)
                vocab_emword_shared = load_vocab(gconfig.vocab_emword_shared)
                save_embedd_lookup(vocab_emword_shared, config_list[0].embedd,
                        gconfig.embedd_lookup_shared, config_list[0].dim_wembedd)
            except IOError:
                print ('Make sure enabled reuse_token, and shared-vocab is %s'%(gconfig.vocab_char_shared))


    """ Training
    Building only one graph for cross-lingual or cross-domain,
    Building data-collector(manage vocabs, embedds and datasets) and
    model-builder(manage session and subgraph) for each treebank or language,
    Global-Graph-Manager(base_model) manage each session

    """
    if gconfig.train and gconfig.data_model == 'seperate-data_seperate-model':
        models = []
        data_collectors = []
        for config in config_list:
            # set data loader for each treebank, load vocabs, embeddings and datasets
            data_loader = DataCollector(config, trainFLAG=True, devFLAG=True)
            data_collectors.append(data_loader)
            # build model
            if gconfig.task == 'parse':
                model = Parser(config, data_loader)
            elif gconfig.task == 'pos':
                model = Tagger(config, data_loader)
            else:
                raise TypeError("Do not support %s!"%(config.task))
            model.build()
            models.append(model)
        # global trainer
        trainer = BaseModel(models)
        trainer.init_session()
        trainer.train_models()

    """ Testing """
    if gconfig.test and gconfig.data_model == 'seperate-data_seperate-model':
        models = []
        data_collectors = []
        for config in config_list:
            # set data loader for each treebank, load vocabs, embeddings and datasets
            data_loader = DataCollector(config, testFLAG=True)
            data_collectors.append(data_loader)
            # build model
            if gconfig.task == 'parse':
                model = Parser(config, data_loader)
            elif gconfig.task == 'pos':
                model = Tagger(config, data_loader)
            else:
                raise TypeError("Do not support %s!"%(config.task))
            model.build()
            models.append(model)
        # global evaluater
        evaluater = BaseModel(models)
        evaluater.init_session()
        evaluater.evaluate_models()
