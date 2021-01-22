# Justin Canode
# 1/12/2021
# Sample code using gensim to vectorize datasets for embedding and testing models
# Using examples from:
# https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-tutorial


# import librarys
from gensim import corpora
import logging
from pprint import pprint  # pretty-printer
from collections import defaultdict
from smart_open import open  # for transparently opening remote files
import unicodedata
from gensim import models
import os
from gensim import similarities
import tempfile


# set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



corpus = corpora.MmCorpus('./tmp/corpus.mm')

document_path = u"C:/Users/justi/documents/RDPTranslation/data/english-data.txt"

# path = u"C:/Users/justi/documents/RDPTranslation/data/english-data.txt"
# # collect statistics about all tokens


dictionary = corpora.Dictionary.load('./tmp/englishDict.dict')

# corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
#     print(doc)
lsi = models.LsiModel(corpus)
corpus_lsi = lsi[corpus]

lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation
# corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi_model.print_topics(2)
# # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
# # for doc, as_text in zip(corpus_lsi, dictionary):
# #     print(doc, as_text)

with tempfile.NamedTemporaryFile(prefix='model-', suffix='.lsi', delete=False) as tmp:
    lsi_model.save(tmp.name)  # same for tfidf, lda, ...

# index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
