import os
import argparse
import copy

from config import *
from utils.dict2obj import Dict2Obj
from models.elmo_test import *

# Location of pretrained LM.  Here we use the test fixtures.
vocab_file = './data/vocab/English/word.vocab' 
options_file = '/home/liyujiang/Project/dataset/elmo/options.json'
weight_file  = '/home/liyujiang/Project/dataset/elmo/weights.hdf5'

if __name__ == '__main__':
    """ Argument Parser Settings """
    """
    # argparser from console
    argparser = argparse.ArgumentParser(prog="Universal Dependencies")
    argparser = argparse_console(argparser)

    # general config object and saving to file
    args = argparser.parse_args()
    args_dict = args.__dict__
    fconfig = './arguments.cfg'
    gconfig = Dict2Obj(args_dict, recur=False)
    gconfig = config_common(gconfig)
    """
    
    elmo_func(vocab_file, options_file, weight_file)
