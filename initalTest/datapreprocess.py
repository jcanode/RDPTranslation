import tensorflow as tf
import time
import io
import os
import numpy as np


## diagnostics to see how long each step takes
start_time = time.time()
## load data
german_data_path = os.path.join(os.path.curdir,"./data/europarl-v7.de-en.de")
english_data_path = os.path.join(os.path.curdir,"./data/europarl-v7.de-en.en")
print("Time required to load data --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
def create_dataset(germ_path, eng_path):
  germ_lines = io.open(germ_path, encoding='UTF-8').read().strip().split('\n')
  eng_lines = io.open(eng_path, encoding='UTF-8').read().strip().split('\n')
  return germ_lines, eng_lines
german_data, english_data = create_dataset(german_data_path, english_data_path)
print("Time required to create dataset --- %s seconds ---" % (time.time() - start_time))

## Tokenize german and english sequences
start_time = time.time()

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
## Uncomment these lines for first time running the program to configure tokenizer (helps speed up run time)
# en_tokenizer.fit_on_texts(english_data)
# english_config = en_tokenizer.to_json()

english_config_path = os.path.join("./tokenizerConfig/english_config.json")
## Uncomment these lines for first time running the program to configure tokenizer (helps speed up run time)
# io.open(english_config_path,'w').write(english_config)
# print("Wrote english config to ", english_config_path)

## Comment lines out first time runnint to avoid reloading config
english_config = io.open(english_config_path, encoding='UTF-8').read()
en_tokenizer =  tf.keras.preprocessing.text.tokenizer_from_json(english_config)
print("Read english tokenizer input from ", english_config_path)

## Tokenize english data
data_en = en_tokenizer.texts_to_sequences(english_data)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

ge_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

## Uncomment these lines for first time running the program to configure tokenizer (helps speed up run time)
# ge_tokenizer.fit_on_texts(german_data)
# german_config = ge_tokenizer.to_json()

german_config_path = os.path.join("./tokenizerConfig/german_config.json")

## Uncomment these lines for first time running the program to configure tokenizer (helps speed up run time)
# io.open(german_config_path,'w').write(german_config)
# print("Wrote german config to ", german_config_path)

## Comment lines out first time runnint to avoid reloading config
german_config = io.open(german_config_path, encoding='UTF-8').read()
ge_tokenizer =  tf.keras.preprocessing.text.tokenizer_from_json(german_config)
print("Read german tokenizer input from ", german_config_path)

## tokenize german data
data_ge = ge_tokenizer.texts_to_sequences(german_data)
data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge,padding='post',maxlen=668) ##pad german dataset to same length as english 

print("Time required to tokenize data --- %s seconds ---" % (time.time() - start_time))


## Write tokenized data
english_data_tokenized_path = os.path.join(os.path.curdir, './data/tokenized/data_en.en')
german_data_tokenized_path = os.path.join(os.path.curdir, './data/tokenized/data_de.de')

data_en.tofile(english_data_tokenized_path,sep="",format="%s")
# io.open(english_data_tokenized_path,'w',encoding='UTF-8').write(data_en)
print("Wrote english tokenized data to %s", english_data_tokenized_path)

data_ge.tofile(german_data_tokenized_path,sep="",format="%s")
# io.open(german_data_tokenized_path, 'w', encoding='UTF-8').write(data_de)
print("Wrote english tokenized data to %s", german_data_tokenized_path)