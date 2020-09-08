import tensorflow as tf
import os
import time
import numpy as np
## Load data
dataset_path = os.path.join("./data/")

dataset = tf.data.experimental.load(dataset_path,element_spec=tf.TensorSpec(shape=(64,334,1)))

print(len(list(dataset)))

# print(list(dataset))
# next(iter(dataset))
# example_X = tf.reshape(example_X,[example_X.shape[0],example_X.shape[1],1])
# example_Y = tf.reshape(example_Y,[example_Y.shape[0],example_Y.shape[1],1])

# print(example_X.shape)
# print(example_Y.shape)

## build model
def buildModel():
    model = tf.keras.Sequential(layers=[
        # tf.keras.layers.LSTM(64),
        # tf.keras.layers.LSTM(16),
        tf.keras.layers.Conv1D(334,kernel_size=1),
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
        tf.keras.layers.Dense(334),

        # tf.keras.layers.Conv1DTranspose(1,kernel_size=1),
        # tf.keras.layers.Conv1DTranspose(668,kernel_size=1)

        ])
    return model


## secondary experimental model
def buildModel2():
    model = tf.keras.Sequential(layers=[
        # tf.keras.layers.LSTM(64),
        # tf.keras.layers.LSTM(16),
        tf.keras.layers.Conv1D(334,kernel_size=1),
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
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(334),
        # tf.keras.layers.Dense(668),
        # tf.keras.layers.Dense(668),
        # tf.keras.layers.Conv1DTranspose(1,kernel_size=1),
        # tf.keras.layers.Conv1DTranspose(668,kernel_size=1)

        ])
    return model
##train model

start_time = time.time()
seq2seq = buildModel()
print("Time required to create model --- %s seconds ---" % (time.time() - start_time))

# seq2seq.summary()

## Train model and print results
seq2seq.compile(optimizer= 'adam',loss="mean_squared_error")
seq2seq.fit(dataset,epochs=100)
# seq2seq.fit(np.random.randn(10,512,16), np.random.randn(10,334),epochs=100)
# seq2seq.save(filepath="./models/seq2seq")
 