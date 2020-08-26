import tensorflow as tf
import tensorflow_datasets as tfds
import time
import  os
import io
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split

## diagnostics to see how long each step takes
start_time = time.time()
## load data
german_data_path = os.path.join(os.path.curdir,"./data/europarl-v7.de-en.de")
english_data_path = os.path.join(os.path.curdir,"./data/europarl-v7.de-en.en")
print("Time required to load data --- %s seconds ---" % (time.time() - start_time))


# english_dataset = []
# english_lines_dataset = tf.data.TextLineDataset(english_data_path)
# # labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
# english_dataset.append(english_lines_dataset)
#
# german_dataset = []
# german_lines_dataset = tf.data.TextLineDataset(german_data_path)
# # labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
# german_dataset.append(german_lines_dataset)
#
# print(german_lines_dataset)

## Load data from file and into a dataset in memory
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
data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge,padding='post')

print("Time required to tokenize data --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
## slice data with an 80/20 split
X_train,  X_test, Y_train, Y_test = train_test_split(data_en,data_ge,test_size=0.8)
print("Time required to slice data --- %s seconds ---" % (time.time() - start_time))

# def maxLength(arr):
#     max = len(arr[0])
#     for i in arr:
#         if (len(arr[i])>max):
#             max=len(arr[i])
#     return arr

# xLength = maxLength(X_train)

# print("max length of sentience: \n" + xLength)

## Convert data to tensor and print shape of X_train (english training data)
Dtype = tf.float32

X_train = tf.convert_to_tensor(X_train,dtype=Dtype)
X_train = tf.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
print(X_train.shape)
X_test = tf.convert_to_tensor(X_test,dtype=Dtype)
Y_train = tf.convert_to_tensor(Y_train,dtype=Dtype)
Y_test = tf.convert_to_tensor(Y_test,dtype=Dtype)



# print(X_train)

## Define model (cnn seq2seq)

#Hyperparams            (from https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt)
BATCH_SIZE = 64
BUFFER_SIZE = 16 #len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32

# input_vocab_size = len(en_tokenizer.word_index)+1
# output_vocab_size = len(ge_tokenizer.word_index)+ 1
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
example_X = tf.reshape(example_X,[example_X.shape[0],example_X.shape[1],1])
# example_X.set_shape([example_X.shape[0],example_X.shape[1],1])
print(example_X.shape)
print(example_Y.shape)

def buildModel():
    model = tf.keras.Sequential(layers=[
        # tf.keras.layers.LSTM(64),
        tf.keras.layers.LSTM(16),
        # tf.keras.layers.Conv1D(64,kernel_size=1), #input_shape=(64,668)
        # tf.keras.layers.MaxPool1D(),
        # tf.keras.layers.Conv1D(16,kernel_size=1),
        tf.keras.layers.Dense(16),
        # # tf.keras.layers.MaxPool1D(),
        #
        tf.keras.layers.Dense(16),
        # tf.keras.layers.Conv1DTranspose(16,kernel_size=4),
        # tf.keras.layers.Conv1DTranspose(64,kernel_size=16)

        ])
    return model


start_time = time.time()
seq2seq = buildModel()
print("Time required to create model --- %s seconds ---" % (time.time() - start_time))

# seq2seq.summary()

## Train model and print results
seq2seq.compile(optimizer= 'adam',loss="categorical_crossentropy")
seq2seq.fit(x=example_X,y=example_X,batch_size=BATCH_SIZE,epochs=10)
seq2seq.save(filepath="./models/seq2seq")
