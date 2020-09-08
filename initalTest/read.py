import tensorflow as tf
import io
import os
import numpy as np
import  time
from sklearn.model_selection import train_test_split

english_data_ path = os.path.join('./data/tokenized/data_en.en')
german_data_path = os.path.join("./data/tokenized/data_de.de")

english_data = np.fromfile(english_data_path)


print("read english data")

english_reshaped_data= np.reshape(english_data,(1920209,334))
print(english_reshaped_data.shape)

print(english_reshaped_data)

german_data = np.fromfile(german_data_path)
print("read german data")
german_reshaped_data=np.reshape(german_data,(1920209,334))
print(german_reshaped_data.shape)

print ("reshaped data")
# start_time = time.time()
## slice data with an 80/20 split
# X_train,  X_test, Y_train, Y_test = train_test_split(english_reshaped_data,german_reshaped_data,test_size=0.7)
# print("Time required to slice data --- %s seconds ---" % (time.time() - start_time))

Dtype = tf.float32

print("converting data to tenosrs")
# X_train = tf.convert_to_tensor(X_train,dtype=Dtype)
# X_train = tf.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
# print(X_train.shape)
# X_test = tf.convert_to_tensor(X_test,dtype=Dtype)

# Y_train = tf.convert_to_tensor(Y_train,dtype=Dtype)
# Y_train = tf.reshape(Y_train, [Y_train.shape[0], Y_train.shape[1], 1])

# Y_test = tf.convert_to_tensor(Y_test,dtype=Dtype)

print("converted data to tensors")

BATCH_SIZE = 64
BUFFER_SIZE = 16 #len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32

print ("loading dataset")
dataset = tf.data.Dataset.from_tensor_slices((english_reshaped_data, german_reshaped_data)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset_path = os.path.join("./data/")
print("saving dataset to %s", dataset_path)
tf.data.experimental.save(dataset,dataset_path)
# writer = tf.data.experimental.TFRecordWriter('mydata.tfrecord')
# writer.write(dataset)

#example_X, example_Y = next(iter(dataset))
#example_X = tf.reshape(example_X,[example_X.shape[0],example_X.shape[1],1])
#example_Y = tf.reshape(example_Y,[example_Y.shape[0],example_Y.shape[1],1])

#print(example_X.shape)
#print(example_Y.shape)
