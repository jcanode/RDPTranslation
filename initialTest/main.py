import tensorflow as tf
# import tensorflow_datasets as tfds
import time
import os
import io
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
from initialTest import datapreprocess


german_data_path, english_data_path = datapreprocess.load_data()

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

german_data, english_data = datapreprocess.create_dataset(german_data_path, english_data_path)
data_en = datapreprocess.english_tokenizer(english_data)
data_ge = datapreprocess.german_tokenizer(german_data)

# TODO: Check code below
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
Y_train = tf.reshape(Y_train, [Y_train.shape[0], Y_train.shape[1], 1])

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
example_Y = tf.reshape(example_Y,[example_Y.shape[0],example_Y.shape[1],1])

# example_X.set_shape([example_X.shape[0],example_X.shape[1],1])
print(example_X.shape)
print(example_Y.shape)

# def buildModel():
#     model = tf.keras.Sequential(layers=[
#         # tf.keras.layers.LSTM(64),
#         tf.keras.layers.LSTM(16),
#         # tf.keras.layers.Conv1D(64,kernel_size=1), #input_shape=(64,668)
#         # tf.keras.layers.MaxPool1D(),
#         # tf.keras.layers.Conv1D(16,kernel_size=1),
#         tf.keras.layers.Dense(16),
#         # # tf.keras.layers.MaxPool1D(),
#         #
#         tf.keras.layers.Dense(16),
#         # tf.keras.layers.Conv1DTranspose(16,kernel_size=4),
#         # tf.keras.layers.Conv1DTranspose(64,kernel_size=16)

#         ])


#     return model


def buildModel():
    model = tf.keras.Sequential(layers=[
        # tf.keras.layers.LSTM(64),
        # tf.keras.layers.LSTM(16),
        tf.keras.layers.Conv1D(668,kernel_size=1),
        tf.keras.layers.Conv1D(128,kernel_size=1),
        tf.keras.layers.Conv1D(64,kernel_size=1),
        tf.keras.layers.Conv1D(1,kernel_size=1),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Conv1D(64,kernel_size=1), #input_shape=(64,668)
        # tf.keras.layers.MaxPool1D(),
        # tf.keras.layers.Conv1D(16,kernel_size=1),
        # tf.keras.layers.Dense(16),
        # # # tf.keras.layers.MaxPool1D(),
        # #
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(668),

        # tf.keras.layers.Conv1DTranspose(1,kernel_size=1),
        # tf.keras.layers.Conv1DTranspose(668,kernel_size=1)

        ])
    return model


## secondary experimental model
def buildModel2():
    model = tf.keras.Sequential(layers=[
        # tf.keras.layers.LSTM(64),
        # tf.keras.layers.LSTM(16),
        # tf.keras.layers.Conv1D(668,kernel_size=1),
        # tf.keras.layers.Conv1D(128,kernel_size=1),
        # tf.keras.layers.Conv1D(64,kernel_size=1),
        # tf.keras.layers.Conv1D(1,kernel_size=1),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Conv1D(64,kernel_size=1), #input_shape=(64,668)
        # tf.keras.layers.MaxPool1D(),
        # tf.keras.layers.Conv1D(16,kernel_size=1),
        # tf.keras.layers.Dense(16),
        # # # tf.keras.layers.MaxPool1D(),
        # #
        tf.keras.layers.Dense(668),
        tf.keras.layers.Dense(668),
        tf.keras.layers.Dense(668),
        tf.keras.layers.Dense(668),
        tf.keras.layers.Dense(668),
        tf.keras.layers.Dense(668),
        # tf.keras.layers.Conv1DTranspose(1,kernel_size=1),
        # tf.keras.layers.Conv1DTranspose(668,kernel_size=1)

        ])
    return model



start_time = time.time()
seq2seq = buildModel()
print("Time required to create model --- %s seconds ---" % (time.time() - start_time))

# seq2seq.summary()

## Train model and print results
seq2seq.compile(optimizer= 'adam',loss="categorical_crossentropy")
seq2seq.fit(x=example_X,y=example_Y,batch_size=BATCH_SIZE,epochs=100)
seq2seq.save(filepath="./models/seq2seq")
