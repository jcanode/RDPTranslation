import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time

# Download the file
# path_to_zip = tf.keras.utils.get_file(
#     'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
#     extract=True)

# path_to_german_file = os.path.curdir+"/../data/europol-v7.de-en.de"

# path_to_english_file = os.path.curdir+"/../data/europol-v7.de-en.en"
# print(path_to_german_file)

# path_to_german_file = "C:/users/justi/Documents/RDPTranslation/initialTest/data/europarl-v7.de-en.de"

# path_to_english_file = "C:/users/justi/Documents/RDPTranslation/initialTest/data/europarl-v7.de-en.en"

# # Justin's Path
# path_to_spanish_file = "E:/RDPTranslation/initialTest/files/europarl-v7.es-en.es"
#
# path_to_english_file = "E:/RDPTranslation/initialTest/files/europarl-v7.es-en.en"

path_to_spanish_file = "C:/Users/User/Documents/GitHub/RDPTranslation/europarl-v7.es-en.es"
path_to_english_file = "C:/Users/User/Documents/GitHub/RDPTranslation/europarl-v7.es-en.en"

print(path_to_spanish_file)

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def join_sentences(english_data, spanish_data):
    span_lines = io.open(spanish_data, encoding='UTF-8').read().strip().split('\n')
    eng_lines = io.open(english_data, encoding='UTF-8').read().strip().split('\n')
    spanish_result = []
    english_result = []
    combined_data = []
    for i in range(1, 30000):  ## only run on first 30k lines
        spanish_lines_clean = preprocess_sentence(span_lines[i])
        english_lines_clean = preprocess_sentence(eng_lines[i])
        combined_data.append(english_lines_clean)
        #         combined_data.append('\t')
        combined_data.append(spanish_lines_clean)
        spanish_result.append(spanish_lines_clean)
        english_result.append(english_lines_clean)
    return combined_data, spanish_result, english_result


data, spanish_data, english_data = join_sentences(path_to_english_file, path_to_spanish_file)

# joined_data = ['\t'.join(x) for x in zip(data[0::2], x[1::2])]
# print(joined_data[0])
# # print(len(joined_data))
# print(english_data[-1])


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')


    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)


    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, dtype='float32', maxlen=179,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    #   targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(english_data)
    target_tensor, targ_lang_tokenizer = tokenize(spanish_data)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


input_tensor, inp_lang_tokenizer = tokenize(english_data)
target_tensor, targ_lang_tokenizer = tokenize(spanish_data)
print(input_tensor.shape)
# print(len(inp_lang_tokenizer))
print(target_tensor.shape)
# print(len(targ_lang_tokenizer))

# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_english_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
# BATCH_SIZE = 64
## for custom model use
BATCH_SIZE = 179
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def buildModel():
    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv1D(179, kernel_size=1),
        tf.keras.layers.Conv1D(64, kernel_size=1),
        tf.keras.layers.Conv1D(64, kernel_size=1),
        tf.keras.layers.Conv1D(64, kernel_size=1),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Conv1DTranspose(16, kernel_size=1),
        tf.keras.layers.Conv1DTranspose(64, kernel_size=1),
        tf.keras.layers.Conv1DTranspose(668, kernel_size=1),
    ])
    return model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# seq2seq.summary()

# with tf.device('/cpu:0'): ## comment out to run on gpu
#     start_time = time.time()
#     seq2seq = buildModel()
#     print("Time required to create model --- %s seconds ---" % (time.time() - start_time))

#     ## Train model and print results
#     seq2seq.compile(optimizer= 'adam',loss=loss_function)
#     seq2seq.build(input_shape=(668,64,1))
#     # seq2seq.summary()
#     start_time = time.time()
#     for (batch, (inp, targ)) in enumerate(dataset.take(32)):
#         inp = tf.reshape(inp,[inp.shape[0],inp.shape[1],1])
#         targ = tf.reshape(targ,[targ.shape[0],targ.shape[1],1])
#         seq2seq.train_on_batch(x=inp,y=targ)
#         print(batch)
#         print("Time taken to train batch: %s", time.time()-start_time)
#     #     print(targ.shape)
#     #     seq2seq.fit(x=input_tensor_train,y=target_tensor_train,epochs=10)
#     seq2seq.summary()
#     seq2seq.save(filepath="./models/dense")
start_time = time.time()
seq2seq = buildModel()
print("Time required to create model --- %s seconds ---" % (time.time() - start_time))
## Train model and print results
seq2seq.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
seq2seq.build(input_shape=(668, 64, 1))
# seq2seq.summary()
start_time = time.time()
for (batch, (inp, targ)) in enumerate(dataset.take(32)):
    inp = tf.reshape(inp, [inp.shape[0], inp.shape[1], 1])
    targ = tf.reshape(targ, [targ.shape[0], targ.shape[1], 1])
    # loss =  seq2seq.train_on_batch(x=inp,y=targ)
    # print(batch)
    # print("Time taken to train batch: %s", time.time()-start_time)
    # print("batch loss: %f", loss)
    #     print(targ.shape)
    seq2seq.fit(x=inp, y=targ, epochs=5, )
#     seq2seq.fit(x=input_tensor_train,y=target_tensor_train,epochs=10)
seq2seq.summary()
# seq2seq.save(filepath="./models/nmt")
es_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='')
es_tokenizer.fit_on_texts(spanish_data)
for (batch, (inp, targ)) in enumerate(dataset.take(16)):
    # print(inp.shape)
    # print(targ.shape)
    inp = tf.reshape(inp, [inp.shape[0], inp.shape[1], 1])
    targ = tf.reshape(targ, [targ.shape[0], targ.shape[1], 1])
    predictions = []
    predictions.append(seq2seq.predict_on_batch(inp))
    german_output = es_tokenizer.sequences_to_texts(predictions[0][batch - 1])
    # print(german_output.shape)
    while ("" in german_output):
        german_output.remove('')
    print(german_output)
    # print(predictions)
# print(predictions[0])
