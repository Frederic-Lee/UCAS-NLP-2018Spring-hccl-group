import numpy as np
import re
import os

from data_process.dataset import UNK, NUM, ROOT, ROOT_DEPREL
from data_process.dataset import EMAIL, AT_SYMBOL, URL


### Processing tokens ###
def token_normalize(token, lowercase=False, uni_digit=False, strong_norm=False, max_len=50):
    """ Normalize the token 
    Args:
        strong_norm: normalize token that is email, url etc.

    """
    numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
    if lowercase:
        token = token.lower()
    if token.isdigit() and uni_digit:
        token = NUM
    if strong_norm:
        token = re.sub(r".+@.+", EMAIL, token)
        token = re.sub(r"@\w+", AT_SYMBOL, token)
        token = re.sub(r"(https?://|www\.).*", URL, token)
    if len(token) > max_len:
        token = token[:max_len]

    return token


def get_token_idx(vocab_token, vocab_emword=None, vocab_char=None, 
        emwordFLAG=False, charFLAG=False, normFLAG=False, unkFLAG=False,
        lemma_replaceFLAG=False):
    """ Convert a token to corresponding index.
    Should be defined before invoking it.
    Args:
        vocab_token: dict[word] = idx where normalized-word-vocab for word
        vocab_emword, emwordFLAG: only for word
        vocab_echar, charFLAG: only for word
        normFLAG: enable normalizing especially for original word, i.e. node.form
        unkGLAG: enable unknown-token replace out-of-vocab token
        lemma_replaceFLAG: replace token for embedding with lemma

    Return:
        token_idx_list:
            e.g. word: (61, 55, [12, 4, 32]) = (word_id, emword_id, list of char ids)

    """
    def func(token, lemma=None):
        """ if token is word, it should be node.form, i.e. original-form """
        # 1. get chars of the original-form-word
        char_ids = []
        if vocab_char is not None and charFLAG == True:
            for char in token:
                # ignore chars out of vocabulary
                if char in vocab_char:
                    char_ids += [vocab_char[char]]

        # 2. normalize word
        if normFLAG:
            token = token_normalize(token, lowercase=True, uni_digit=True, strong_norm=True)

        # 3. get idx of word for embedd-lookup
        emword_idx = None
        if vocab_emword is not None and emwordFLAG == True:
            if token in vocab_emword:
                emword_idx = vocab_emword[token]
            elif lemma_replaceFLAG is True and lemma is not None and lemma in vocab_emword:
                emword_idx = vocab_emword[lemma]
            else:
                if unkFLAG:
                    emword_idx = vocab_emword[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 4. get idx of token for ngram
        token_idx = None
        if vocab_token is not None:
            if token in vocab_token:
                token_idx = vocab_token[token]
            else:
                if unkFLAG:
                    token_idx = vocab_token[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 5. return tuple token_idx, emword_idx, char_ids
        return token_idx, emword_idx, char_ids

    return func


### Padding sequence ###
def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a sequence list where each sublist has same length,
        a list of real length without padding.

    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length-len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length

    """
    # word and tag processing
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)
    # character processing
    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


### Chunking Functions ###
def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
