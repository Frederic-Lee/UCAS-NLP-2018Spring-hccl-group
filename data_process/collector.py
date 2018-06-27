import os

from data_process.vocab import load_embedd_lookup, load_vocab
from data_process.data_utils import get_token_idx
from data_process.dataset import ConllDataset


class DataCollector():
    def __init__(self, config, trainFLAG=False, devFLAG=False, testFLAG=False):
        """ Collection of vocabularys, embeddings and datasets(conllDataset)

        Supposing data has been bulit successfully enabled by 'build_data' and
        corresponding files have been created (vocab and lookedup-embeddings)

        """
        # 1. load vocabulary
        if config.reuse_subtoken:
            self.vocab_char = load_vocab(config.vocab_char_shared)
        else:
            self.vocab_char = load_vocab(config.vocab_char)

        if config.reuse_token:
            self.vocab_word   = load_vocab(config.vocab_emword_shared)
            self.vocab_emword = load_vocab(config.vocab_emword_shared)
        else:
            self.vocab_word   = load_vocab(config.vocab_emword) # can be vocab_emword or vocab_word
            self.vocab_emword = load_vocab(config.vocab_emword)

        self.vocab_upos   = load_vocab(config.vocab_upos)
        self.vocab_xpos   = load_vocab(config.vocab_xpos)
        self.vocab_lemma  = load_vocab(config.vocab_lemma)
        self.vocab_deprel = load_vocab(config.vocab_deprel)

        self.nword   = len(self.vocab_word['str2idx'])
        self.nchar   = len(self.vocab_char['str2idx'])
        self.nupos   = len(self.vocab_upos['str2idx'])
        self.nxpos   = len(self.vocab_xpos['str2idx'])
        self.ndeprel = len(self.vocab_deprel['str2idx'])
        self.nemword = len(self.vocab_emword['str2idx'])

        # 2. get functions mapping str -> id
        self.token2idx_funcs = dict()
        self.token2idx_funcs['word']  = get_token_idx(self.vocab_word['str2idx'],
                self.vocab_emword['str2idx'], self.vocab_char['str2idx'],
                emwordFLAG=True, charFLAG=True, normFLAG=True, unkFLAG=True, lemma_replaceFLAG=True)
        self.token2idx_funcs['upos']  = get_token_idx(self.vocab_upos['str2idx'])
        self.token2idx_funcs['xpos']  = get_token_idx(self.vocab_xpos['str2idx'])
        self.token2idx_funcs['deprel'] = get_token_idx(self.vocab_deprel['str2idx'])

        if config.seqlabFLAG:
            self.vocab_seqlab = load_vocab(config.vocab_seqlab)
            self.nseqlab = len(self.vocab_seqlab['str2idx'])
            self.token2idx_funcs['seqlab'] = get_token_idx(self.vocab_seqlab['str2idx'])

        # 3. get pre-trained embeddings
        if config.reuse_token:
            self.embeddings = (load_embedd_lookup(config.embedd_lookup_shared)
                    if config.wembeddFLAG else None)
        else:
            self.embeddings = (load_embedd_lookup(config.embedd_lookup)
                    if config.wembeddFLAG else None)

        # 4. load conll-dataset
        self.train = None
        self.dev   = None
        self.test  = None
        if trainFLAG:
            self.train = ConllDataset(config.train_file, convert2idx=True, token2idx_funcs=self.token2idx_funcs, enable_seqlabel=config.seqlabFLAG, count_seq=True)
        if devFLAG:
            self.dev   = ConllDataset(config.dev_file, convert2idx=True, token2idx_funcs=self.token2idx_funcs, enable_seqlabel=config.seqlabFLAG, count_seq=True)
        if testFLAG:
            self.test  = ConllDataset(config.test_file, convert2idx=True, token2idx_funcs=self.token2idx_funcs, enable_seqlabel=config.seqlabFLAG, count_seq=True)
