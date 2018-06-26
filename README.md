# UCAS-NLP-2018Spring-hccl-group
Course Project of Natural Language Processing, focused on Tagging and Dependency Parsing.

Dataset:
Universal Dependencies v2.1
https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2515/ud-treebanks-v2.1.tgz?sequence=4&isAllowed=y
UD uses a revised version of the CoNLL-X format called CoNLL-U. Annotations are encoded in plain text files (UTF-8, using only the LF character as line break, including an LF character at the end of file) with three types of lines:
  1.Word lines containing the annotation of a word/token in 10 fields separated by single tab characters; see below.
  2.Blank lines marking sentence boundaries.
  3.Comment lines starting with hash (#).
Sentences consist of one or more word lines, and word lines contain the following fields:
  1.ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
  2.FORM: Word form or punctuation symbol.
  3.LEMMA: Lemma or stem of word form.
  4.UPOS: Universal part-of-speech tag.
  5.XPOS: Language-specific part-of-speech tag; underscore if not available.  
  6.FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
  7.HEAD: Head of the current word, which is either a value of ID or zero (0).
  8.DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
  9.DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
  10.MISC: Any other annotation.
Detailed information is available in the officail website:
http://universaldependencies.org/

Pre-trained Embeddings:
FastText
https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
Pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia using fastText. These models were trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.
For English, we suggest Glove embeddings trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download)
http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
ELMo in English is also supported in this project. We have implemented "computing representations on the fly from raw text using character input". Please download the options file and weights file in the following website:
https://github.com/allenai/bilm-tf
