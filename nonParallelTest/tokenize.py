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

# set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




path = u"C:/Users/User/Documents/RDP2020/NMT/RDPTranslation/data/europarl-v7.de-en.en"
# collect statistics about all tokens


dictionary = corpora.Dictionary(line.lower().split() for line in open(path, encoding="utf8"))

texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in open(path,encoding="utf8")
]

# remove stop words and words that appear only once
stoplist = set('for a of the and to in'.split())

stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
dictionary.save('./tmp/englishDict.dict')
print(dictionary)
#


class MyCorpus:
    def __iter__(self):
        for line in open(path):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)