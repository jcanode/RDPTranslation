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

tfidf_corpus = corpora.MmCorpus('./tmp/corpus.mm')

# document_path = u"C:/Users/justi/documents/RDPTranslation/data/english-data.txt"
dictionary = corpora.Dictionary.load('./tmp/englishDict.dict')


tmp_file= "C:/Users/justi/AppData/Local/Temp/model-8d52k3xq.lsi"
# with tempfile.NamedTemporaryFile(prefix='model-', suffix='.lsi', delete=False) as tmp:
#     loaded_lsi_model = models.LsiModel.load(tmp.name)

    # lsi_model.save(tmp.name)  # same for tfidf, lda, ...

loaded_lsi_model = models.LsiModel.load(tmp_file)

model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)

gov_answers = model.most_similar('government')
print(gov_answers)