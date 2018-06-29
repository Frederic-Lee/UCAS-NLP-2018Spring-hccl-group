import os
import numpy as np
from math import ceil

# Shared global variables to be imported from model
UNK = "<UNK>"
NUM = "<NUM>"
NONE = "_"
ROOT = "<ROOT>"
ROOT_DEPREL = "<RROOT>"
# special tokens substitue unknown-token
EMAIL = "<EMAIL>"
URL = "<URL>"
AT_SYMBOL = "<AT>"

from data_process.data_utils import token_normalize, pad_sequences


class ConllNode:
    """ Each Line for a token in UD-Treebanks """
    def __init__(self, idx, form, lemma=None, upos=None, xpos=None,
            feats=None, head=None, deprel=None, deps=None, misc=None):
        """
        INPUT:
            --IDX: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; \
            may be a decimal number for empty nodes.
            --FORM: Word form or punctuation symbol in original form, which may be used for character.
            --LEMMA: Lemma or stem of word form.
            --UPOS: Universal part-of-speech tag, which is benefitcial to embedding.
            --XPOS: Language-specific part-of-speech tag; underscore if not available.
            --FEATS: List of morphological features from the universal feature inventory or from a defined language-specific /
            extension; underscore if not available.
            --HEAD: Head of the current word, which is either a value of ID or zero (0).
            --DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
            --DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
            --MISC: Any other annotation.

        """
        self.idx  = idx
        self.form = form
        self.norm = token_normalize(form, lowercase=True, uni_digit=True, strong_norm=True)
        self.upos = upos.upper()
        self.xpos = xpos.upper()
        self.head = head
        self.deprel = deprel

        self.lemma = lemma
        self.feats = feats
        self.deps  = deps
        self.misc  = misc

        self.pred_head = None
        self.pred_deprel = None
        self.pred_upos = None
        self.pred_xpos = None

    def set_head_rel(self, head, deprel):
        self.pred_head = head
        self.pred_deprel = deprel

    def set_pos(self, upos, xpos):
        self.pred_upos = upos
        self.pred_xpos = xpos

    def tostr(self, predict=True):
        if predict is not True:
            values = [str(self.idx), self.form, self.lemma, self.upos, self.xpos, self.feats,
                    str(self.head), self.deprel, self.deps, self.misc]
        elif self.pred_head is not None and self.pred_deprel is not None:
            values = [str(self.idx), self.form, self.lemma, self.upos, self.xpos, self.feats,
                    str(self.pred_head), self.pred_deprel, self.deps, self.misc]
        elif self.pred_xpos is not None and self.pred_xpos is not None:
            values = [str(self.idx), self.form, self.lemma, self.pred_upos, self.pred_xpos, self.feats,
                    str(self.head), self.deprel, self.deps, self.misc]
        else:
            raise ValueError("Writting to file failed!")

        return '\t'.join([NONE if v is None else v for v in values])


class ConllDataset(object):
    """Class that iterates over CoNLL-format Dataset

    iter_datasets method yields a list of nodes of the sentence
    iter_batches method yields a dictionary of token-idx

    Example:
        data = CoNLLDataset(filename)
        for sentence in data.iter_datasets():
            pass
        for batch in data.iter_batches():
            pass

    """
    def __init__(self, filename, convert2idx=False, token2idx_funcs=None, enable_seqlabel=False, count_seq=False):
        """
        Args:
            --filename: path to the dataset
            --convert2idx: whether convert each token to idx
            --token2idx_funcs: list of 'get_token_idx' functions for different types of tokens
            --enable_seqlabel: dataset has sequence-label for each sequence
            --count_seq: whether count total sequences in the dataset

        """
        self.filename = filename
        self.convert2idx = convert2idx
        self.func_list = token2idx_funcs
        self.buckets = [5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]
        self.enable_seqlabel = enable_seqlabel
        self.nseq = self._count_seq() if count_seq is True else 0


    def iter_dataset(self):
        """ Iter CoNLL-FORMAT Dataset
        Iteratively return a list of nodes for each sentence in the conll-dataset.
        Add ROOT-node to facilitate decoding, drop the root by nodes[1:].

        Returns:
            list: [sent_len] * ConllNode(10 types)
            or tuple if has seqlabel: ([sent_len] * ConllNode(10 types), seqlab)

        """
        root = ConllNode(0, ROOT, ROOT, ROOT, ROOT, NONE, 0, ROOT_DEPREL, NONE, NONE)
        tokens = [root]
        seqlabel = None
        with open(self.filename) as dataset:
            for line in dataset:
                tok = line.strip().split('\t')
                if not tok or line.strip() == '':
                    if len(tokens)>1:
                        yield tokens if self.enable_seqlabel is False else (tokens, seqlabel)
                    tokens = [root]
                else:
                    if line[0] == '#' or '.' in tok[0]:
                        lucky = 1
                        seqlabel = line.strip().split(' ')[-1]
                    elif line[0] == '#' and '-label:' in tok[1]:
                        seqlabel = tok[2]
                    elif '-' in tok[0]:
                        # for multi-word
                        print (self.filename)
                        print ("multi-word idx: %s"%(tok))
                        start, end = map(int, tok[0].split('-'))
                        for _ in range(start, end+1):
                            line = dataset.readline()
                            tok  = line.strip().split('\t')
                            node = ConllNode(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5],
                                int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])
                            tokens.append(node)
                    else:
                        node = ConllNode(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5],
                                int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])
                        tokens.append(node)

            if len(tokens) > 1:
                yield tokens if self.enable_seqlabel is False else (tokens, seqlabel)


    def _count_seq(self):
        """Count sequences in dataset """
        total_seq = 0
        generator = self.iter_dataset()
        for idx, _ in enumerate(generator):
            total_seq = idx + 1
        print("%d sequences in %s..."%(total_seq, self.filename.split('/')[-1]))

        return total_seq


    def iter_batch(self, batch_size, shuffle=True):
        """
        Args:
            self.iter_dataset: iterable dataloader yield dictionary of idx-list
            batch_size: (int)
            shuffle: if True, reload full dataset then split, else iterablly output.
        Yields:
            list of dictionary: [batch_sz, dict(word-ids, char-ids, emword-ids,
            upos-ids, xpos-ids, head-ids, deprel-ids)]
            list of nodes: [batch_sz, node_list]
            seq-labels: [batch_sz]

        """
        def no_shuffle_iter():
            """ Iterably reading batch of dataset in order and padding """
            batch = []
            nodes = []
            seqlabs = []
            for tokens in self.iter_dataset():
                if len(batch) == batch_size:
                    # yield padded batch
                    yield (self._pad_batch(batch), nodes) if self.enable_seqlabel is False else (
                            self._pad_batch(batch), nodes, seqlabs)
                    batch = []
                    nodes = []
                    seqlabs = []
                # convert to idx
                if self.enable_seqlabel is True:
                    tokens, seqlab = tokens
                    seqlab_id = self.func_list['seqlab'](seqlab)[0]
                    seqlabs.append(seqlab_id)
                token_ids = self._get_token_idx_dict(tokens)
                batch.append(token_ids)
                nodes.append(tokens)
            if len(batch) != 0:
                yield (self._pad_batch(batch), nodes) if self.enable_seqlabel is False else (
                        self._pad_batch(batch), nodes, seqlabs)

        def shuffle_iter():
            """ Reading dataset into buckets and iter dataset """
            # reload full-dataset nodes
            full_dataset = [[] for _ in self.buckets]
            for tokens in self.iter_dataset():
                if self.enable_seqlabel:
                    tokens, seqlab = tokens
                if len(tokens)-1 > self.buckets[-1]:
                    raise ValueError("Exceed max-bucket size with %d!"%(len(tokens)-1))
                for bucket_id, bucket_sz in enumerate(self.buckets):
                    if len(tokens)-1 <= bucket_sz:
                        if self.enable_seqlabel:
                            tokens = (tokens, seqlab)
                        full_dataset[bucket_id].append(tokens)
                        break
            # shuffle dataset
            while [] in full_dataset:
                full_dataset.remove([])
            np.random.shuffle(full_dataset)
            for bucket_id in range(len(full_dataset)):
                np.random.shuffle(full_dataset[bucket_id])
            # yield padded batch
            for bucket_data in full_dataset:
                for start_idx in range(0, len(bucket_data), batch_size):
                    batch_nodes = bucket_data[start_idx:start_idx+batch_size]
                    if self.enable_seqlabel:
                        batch_nodes, seqlabs = batch_nodes
                        seqlab_ids = [self.func_list['seqlab'](seqlab)[0] for seqlab in seqlabs]
                    batch_ids = [self._get_token_idx_dict(tokens) for tokens in batch_nodes]
                    yield (self._pad_batch(batch_ids), batch_nodes) if self.enable_seqlabel is False else (
                            self._pad_batch(batch_ids), batch_nodes, seqlab_ids)

        if shuffle is True:
            return shuffle_iter()
        else:
            return no_shuffle_iter()


    def _pad_batch(self, batch, pad_tok=0):
        """Pad a batch of idx-lists
        Args:
            batch: list of dictionarys consist of idx-lists
            pad_tok: padding token
        Returns:
            feed_dict: sequence_lengths, word_lengths, token-idx-lists
                [batch_sz, max_seq_len] for tokens
                [batch_sz, max_seq_len, max_word_len] for subtokens

        """
        # list of each ype of tokens [batch_sz, individual_len_sent]
        word_ids = [sent_dict['word'] for sent_dict in batch]
        char_ids = [sent_dict['char'] for sent_dict in batch]
        upos_ids = [sent_dict['upos'] for sent_dict in batch]
        xpos_ids = [sent_dict['xpos'] for sent_dict in batch]
        head_ids = [sent_dict['head'] for sent_dict in batch]
        deprel_ids = [sent_dict['deprel'] for sent_dict in batch]
        emword_ids = [sent_dict['emword'] for sent_dict in batch]
        # pad each type of tokens
        word_feed, sequence_lengths = pad_sequences(word_ids, pad_tok)
        char_feed, word_lengths     = pad_sequences(char_ids, pad_tok, nlevels=2)
        upos_feed, _ = pad_sequences(upos_ids, pad_tok)
        xpos_feed, _ = pad_sequences(xpos_ids, pad_tok)
        head_feed, _ = pad_sequences(head_ids, pad_tok)
        deprel_feed, _ = pad_sequences(deprel_ids, pad_tok)
        emword_feed, _ = pad_sequences(emword_ids, pad_tok)
        # feed_dictionary
        feed_dict = {'sequence_lengths':sequence_lengths, 'word_lengths':word_lengths,
                'word':word_feed, 'char':char_feed, 'emword':emword_feed,
                'upos':upos_feed, 'xpos':xpos_feed, 'deprel':deprel_feed, 'head':head_feed}

        return feed_dict


    def _get_token_idx_dict(self, data):
        """ Convert tokens to corresponding idxs given pre-defined functions in func_list.
        Args:
            func_list: list of pre-defined get_token_idx functions
            data: list of ConllNodes iterabled by ConllDataset
        Returns:
            dictionary of idx-lists of several type of tokens,
            [word, emword, char, upos, xpos, head, deprel]

        """
        word_ids = []
        char_ids = []
        emword_ids = []
        upos_ids = []
        xpos_ids = []
        head_ids = []
        deprel_ids = []
        for node in data:
            word, emword, chars = self.func_list['word'](node.form, node.lemma)
            upos, _, _  = self.func_list['upos'](node.upos)
            xpos, _, _  = self.func_list['xpos'](node.xpos)
            deprel, _, _ = self.func_list['deprel'](node.deprel)
            head = int(node.head)
            word_ids.append(word)
            char_ids.append(chars)
            emword_ids.append(emword)
            upos_ids.append(upos)
            xpos_ids.append(xpos)
            head_ids.append(head)
            deprel_ids.append(deprel)

        dic = dict()
        dic['word'] = word_ids
        dic['emword'] = emword_ids
        dic['char'] = char_ids
        dic['upos'] = upos_ids
        dic['xpos'] = xpos_ids
        dic['head'] = head_ids
        dic['deprel'] = deprel_ids

        return dic

    def write_dataset(self, filename, data, append=False, pred=True, have_root=False):
        """ Write date into file in CoNLL-format.
        Args:
            data: token not in idx, including root.
            not write root into the file.

        """
        if append is True:
            open_format = 'a'
        else:
            open_format = 'w'

        with open(filename, open_format) as f:
            for sent in data:
                if have_root:
                    sent = sent[1:]
                for node in sent:
                    f.write(node.tostr(pred)+'\n')
                f.write('\n')
