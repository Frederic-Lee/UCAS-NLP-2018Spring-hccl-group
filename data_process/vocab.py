import numpy as np
import os

from data_process.dataset import UNK, NUM, NONE, ROOT, ROOT_DEPREL
from data_process.dataset import EMAIL, URL, AT_SYMBOL
from data_process.dataset import ConllNode

### Vocabulary Functions ###
def add_special_tokens(vocab, special_tokens):
    """Add special tokens into the vocab"""
    for token in special_tokens:
        vocab.add(token)


def get_token_vocab(conllDatasets, seqlabFLAG=False):
    """ Get different-format token-vocab in UD-Treebank format.
    Args:
        conllDataset: list of iterable ConllDataset Object for a treebank, e.g. [train, dev]
    Return:
        vocabulary for normalized_word, character, lemma, upos, xpos, deprel

    """
    vocab_word = set()
    vocab_char = set()
    vocab_upos = set()
    vocab_xpos = set()
    vocab_lemma  = set()
    vocab_deprel = set()
    vocab_seqlab = set()

    print("Reading dataset ...")
    for dataset in conllDatasets:
        for sentence in dataset.iter_dataset():
            if seqlabFLAG:
                sentence, seqlab = sentence
                vocab_seqlab.add(seqlab)
            vocab_word.update([node.norm for node in sentence if isinstance(node, ConllNode)])
            vocab_upos.update([node.upos for node in sentence if isinstance(node, ConllNode)])
            vocab_xpos.update([node.xpos for node in sentence if isinstance(node, ConllNode)])
            vocab_lemma.update([node.lemma for node in sentence if isinstance(node, ConllNode)])
            vocab_deprel.update([node.deprel for node in sentence if isinstance(node, ConllNode)])
            # special tokens: NUM, ROOT, ROOT_DEPREL have already added into dicts if add root.

            sent_words = [node.form for node in sentence if isinstance(node, ConllNode)]
            for word in sent_words:
                vocab_char.update(word)

    add_special_tokens(vocab_word, [UNK])
    add_special_tokens(vocab_lemma, [UNK])

    return vocab_word, vocab_char, vocab_lemma, vocab_upos, vocab_xpos, vocab_deprel, vocab_seqlab


def get_embedd_vocab(filename):
    """Load vocab from file
    Args:
        filename: path to the embeddings
        vocab: set() of strings

    """
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)

    add_special_tokens(vocab, [UNK, NUM, EMAIL, AT_SYMBOL, URL])

    return vocab


def save_vocab(vocab, filename):
    """Save a vocab by dictionary-sort, one word per line.
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        vocab = list(vocab)
        vocab.sort()
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. %d tokens written to %s" % (len(vocab), filename.split('/')[-1]))


def load_vocab(filename):
    """Loads vocab from a file
    Args:
        filename: (string) the format of the file must be one word per line.
    Returns:
        vocab_dict: vocab_dict['str2idx'] = str2idx[token] = index
                    vocab_dict['idx2str'] = idx2str[index] = token

    """
    try:
        str2idx = dict()
        idx2str = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                str2idx[word] = idx
                idx2str[idx]  = word
        vocab_dict = dict()
        vocab_dict['str2idx'] = str2idx
        vocab_dict['idx2str'] = idx2str
    except IOError:
        print ('Make sure you have already built datasets and vocabularys!')

    return vocab_dict


def union_vocab(name, config_list, out_dir):
    """ Union multiple vocabs and save it.
    Args:
        name: 'char', 'word', 'emword'
        config_list: list of treebank-specific configs, contain vocab-dir

    """
    token_set = set()
    for config in config_list:
        if name == 'char':
            vocab = config.vocab_char
        if name == 'word':
            vocab = config.vocab_word
        if name == 'emword':
            vocab = config.vocab_emword
        with open(vocab, 'r') as f:
            for token in f:
                token_set.add(token.strip())

    save_vocab(token_set, out_dir)


### Embedding Functions ###
def save_embedd_lookup(vocab, embedd_filename, saving_filename, dim):
    """Save embeddings to be lookedup in the numpy array
    Args:
        vocab: emword dictionary, vocab[word] = index
        embedd_filename: a path to a embedd file
        saving_filename: a path where to store a matrix in numpy-array
        dim: (int) dimension of embeddings

    """
    print ("Writting embeddings to be looked-up...")
    vocab = vocab['str2idx']
    embeddings = np.zeros([len(vocab), dim])
    with open(embedd_filename) as f:
        first_line = f.readline().strip().split(' ')
        print (" - %s tokens, %s-dimension..." % (first_line[0], first_line[1]))
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(saving_filename, embeddings=embeddings)
    print (" - done. %d embeddings written to %s." % (len(vocab),
        saving_filename.split('/')[-1]))


def load_embedd_lookup(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]
    except IOError:
        print ('Can not find lookedup embeddings, make sure you have built data.')
