import tensorflow as tf
import time
import io
import os


def load_data():
    # diagnostics to see how long each step takes
    start_time = time.time()
    # Justin's Path
    # german_data_path = os.path.join(os.path.curdir, "./data/europarl-v7.de-en.de")
    # english_data_path = os.path.join(os.path.curdir, "./data/europarl-v7.de-en.en")
    # Emilee's Path
    german_data_path = os.path.join(os.path.curdir,
                                    "C:/Users/User/Documents/GitHub/RDPTranslation/europarl-v7.de-en.de")
    english_data_path = os.path.join(os.path.curdir,
                                     "C:/Users/User/Documents/GitHub/RDPTranslation/europarl-v7.de-en.en")
    print("Time required to load data --- %s seconds ---" % (time.time() - start_time))
    return german_data_path, english_data_path


def create_dataset(germ_path, eng_path):
    start_time = time.time()
    germ_lines = io.open(germ_path, encoding='UTF-8').read().strip().split('\n')
    eng_lines = io.open(eng_path, encoding='UTF-8').read().strip().split('\n')
    print("Time required to create dataset --- %s seconds ---" % (time.time() - start_time))
    return germ_lines, eng_lines


def english_tokenizer_first_time(english_data):
    start_time = time.time()
    en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    en_tokenizer.fit_on_texts(english_data)
    english_config = en_tokenizer.to_json()
    # Justin's Path
    english_config_path = os.path.join("./tokenizerConfig/english_config.json")
    # Emilee's Path
    english_config_path = os.path.join(
      "C:/Users/User/Documents/GitHub/RDPTranslation/tokenizerConfig/english_config.json")
    io.open(english_config_path, 'w').write(english_config)
    print("Wrote english config to ", english_config_path)
    data_en = en_tokenizer.texts_to_sequences(english_data)
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')
    print("Time required to tokenize English data --- %s seconds ---" % (time.time() - start_time))
    return data_en


def english_tokenizer(english_data):
    start_time = time.time()
    en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    # Justin's Path
    # english_config_path = os.path.join("./tokenizerConfig/english_config.json")
    # Emilee's Path
    english_config_path = os.path.join(
      "C:/Users/User/Documents/GitHub/RDPTranslation/tokenizerConfig/english_config.json")
    english_config = io.open(english_config_path, encoding='UTF-8').read()
    en_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(english_config)
    print("Read english tokenizer input from ", english_config_path)
    data_en = en_tokenizer.texts_to_sequences(english_data)
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')
    print("Time required to tokenize English data --- %s seconds ---" % (time.time() - start_time))
    return data_en


def german_tokenizer_first_time(german_data):
    start_time = time.time()
    ge_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    ge_tokenizer.fit_on_texts(german_data)
    german_config = ge_tokenizer.to_json()
    # Justin's Path
    # german_config_path = os.path.join("./tokenizerConfig/german_config.json")
    # Emilee's Path
    german_config_path = os.path.join(
      "C:/Users/User/Documents/GitHub/RDPTranslation/tokenizerConfig/german_config.json")
    io.open(german_config_path,'w').write(german_config)
    print("Wrote german config to ", german_config_path)
    data_ge = ge_tokenizer.texts_to_sequences(german_data)
    data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge, padding='post', maxlen=668)
    print("Time required to tokenize German data --- %s seconds ---" % (time.time() - start_time))
    return data_ge


def german_tokenizer(german_data):
    start_time = time.time()
    ge_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    # Justin's Path
    # german_config_path = os.path.join("./tokenizerConfig/german_config.json")
    # Emilee's Path
    german_config_path = os.path.join(
      "C:/Users/User/Documents/GitHub/RDPTranslation/tokenizerConfig/german_config.json")
    german_config = io.open(german_config_path, encoding='UTF-8').read()
    ge_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(german_config)
    print("Read german tokenizer input from ", german_config_path)
    data_ge = ge_tokenizer.texts_to_sequences(german_data)
    data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge, padding='post', maxlen=668)  # pad ge to en length
    print("Time required to tokenize German data --- %s seconds ---" % (time.time() - start_time))
    return data_ge


def write_tokenized_data(data_en, data_ge):
    # Justin's Path
    # english_data_tokenized_path = os.path.join(os.path.curdir, './data/tokenized/data_en.en')
    # german_data_tokenized_path = os.path.join(os.path.curdir, './data/tokenized/data_de.de')
    # Emilee's Path
    english_data_tokenized_path = os.path.join(
        os.path.curdir, 'C:/Users/User/Documents/GitHub/RDPTranslation/data/tokenized/data_en.en')
    german_data_tokenized_path = os.path.join(
        os.path.curdir, 'C:/Users/User/Documents/GitHub/RDPTranslation/data/tokenized/data_de.de')
    data_en.tofile(english_data_tokenized_path, sep="", format="%s")
    #  This was commented originally
    #  io.open(english_data_tokenized_path, 'w', encoding='UTF-8').write(data_en)
    print("Wrote english tokenized data to %s", english_data_tokenized_path)
    data_ge.tofile(german_data_tokenized_path, sep="", format="%s")
    #  io.open(german_data_tokenized_path, 'w', encoding='UTF-8').write(data_de)
    print("Wrote german tokenized data to %s", german_data_tokenized_path)


def main_first_time():
    german_data_path, english_data_path = load_data()
    german_data, english_data = create_dataset(german_data_path, english_data_path)
    data_en = english_tokenizer_first_time(english_data)
    data_ge = german_tokenizer_first_time(german_data)
    write_tokenized_data(data_en, data_ge)


def main():
    german_data_path, english_data_path = load_data()
    german_data, english_data = create_dataset(german_data_path, english_data_path)
    data_en = english_tokenizer(english_data)
    data_ge = german_tokenizer(german_data)
    write_tokenized_data(data_en, data_ge)
