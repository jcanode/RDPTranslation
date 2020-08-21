import tensorflow as tf
import tensorflow_datasets as tfds
import  os
import io
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split

## load data
german_data_path = os.path.join(os.path.curdir,"./data/europarl-v7.de-en.de")
english_data_path = os.path.join(os.path.curdir,"./data/europarl-v7.de-en.en")


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

def create_dataset(germ_path, eng_path):
  germ_lines = io.open(germ_path, encoding='UTF-8').read().strip().split('\n')
  eng_lines = io.open(eng_path, encoding='UTF-8').read().strip().split('\n')
  return germ_lines, eng_lines
german_data, english_data = create_dataset(german_data_path, english_data_path)

# german_data_test = german_data[int(.5*len(german_data)):]
# english_data_test = english_data[int(.5*len(english_data)):]

# print()

# for i in range(1,11):
#     print("English: \n", english_data[i])
#     print("German: \n", german_data[i])

# split data

#toikenizers

## commented out since they were crashing the program

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(english_data)

data_en = en_tokenizer.texts_to_sequences(english_data)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

ge_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
ge_tokenizer.fit_on_texts(german_data)

data_ge = ge_tokenizer.texts_to_sequences(german_data)
data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge,padding='post')



# slice data with an 80/20 split
X_train,  X_test, Y_train, Y_test = train_test_split(english_data,german_data,test_size=0.6)

# X_train =

## Define model (cnn seq2seq)

#Hyperparams            (from https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt)
BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32

# input_vocab_size = len(en_tokenizer.word_index)+1
# output_vocab_size = len(ge_tokenizer.word_index)+ 1
# dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# example_X, example_Y = next(iter(dataset))
# #print(example_X.shape)
# #print(example_Y.shape)

def buildModel():
    # inputs = tf.keras.Input(shape=(64,16))

    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv1D(4,kernel_size=16,input_shape=(None,1)),
        tf.keras.layers.Conv1D(16,kernel_size=4),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Conv1DTranspose(16,kernel_size=4),
        tf.keras.layers.Conv1DTranspose(64,kernel_size=16)])
    # model.build(input_shape=tf.keras.Input(shape=(1,64,16)))

    return model

seq2seq = buildModel()
# seq2seq = seq2seq(inputs)
seq2seq.summary()

# Train model and print results
seq2seq.compile(optimizer= 'adam',loss="binary_crossentropy")
seq2seq.fit(x=X_train,y=Y_train,batch_size=BATCH_SIZE,epochs=10)
seq2seq.save(filepath="./models/seq2seq")
