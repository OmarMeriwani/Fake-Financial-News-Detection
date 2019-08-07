import numpy as np
from string import punctuation
from os import listdir
import pandas as pd
from numpy import zeros
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn.model_selection import train_test_split
from nltk.stem.porter import *
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from Vocabulary import clean_doc

from keras.layers import LSTM
from keras.layers import Bidirectional
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import RepeatVector
import os
from stanfordcorenlp import StanfordCoreNLP

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host = 'http://localhost'
port = 9000
scnlp = StanfordCoreNLP(host, port=port, lang='en', timeout=30000)

stemmer = PorterStemmer()
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def doc_to_clean_lines(doc, vocab):
    clean_lines = ''
    lines = doc.splitlines()
    for line in lines:
        clean_lines = ' '.join(clean_doc(line))
    return clean_lines


# load the vocabulary
vocab_filename = 'vocabulary.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
vocab = [v.lower() for v in vocab]


def get_weight_matrix2(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size2 = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size2, 300))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab:
        vector = None
        try:
            vector = embedding.get_vector(word)
        except:
            continue
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


def readfile(filename):
    df = pd.read_csv(filename, header=0)
    mode = 'sentence'
    data = pd.DataFrame(columns=['title', 'effect'])
    prev = ''
    seq = 0
    for i in range(0, len(df)):
        sentence = df.loc[i][3]
        NER = scnlp.ner(sentence)
        # sentence = ' '.join([w for w, n in NER if n == 'O'])
        sentenceList = []
        for w, n in NER:
            # print(w, n)
            if str(n) == 'O':
                sentenceList.append(w)
            else:
                sentenceList.append('NER')
        sentence = ' '.join(sentenceList)
        effect = df.loc[i][4]
        # print(df.loc[i][4])
        if effect > 0:
            effect = 1
        else:
            effect = -1
        sentence = doc_to_clean_lines(sentence, vocab)
        if sentence.strip() != '':
            data.loc[seq] = [sentence, effect]
            print(sentence, effect)
            seq += 1
    return data


data = readfile('SSIX News headlines Gold Standard EN.csv')
headlines = data[['title']]
effects = data[['effect']]

x_train, x_test, y_train, y_test = train_test_split(headlines, effects, test_size=0.2)
traindata = np.array(x_train)
testdata = np.array(x_test)

train_docs = traindata[:, 0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

# define training labels
test_docs = testdata[:, 0]
'''==============================================='''
# pad sequences
encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

encoded_docs = tokenizer.texts_to_sequences(test_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
wv_from_bin = KeyedVectors.load_word2vec_format(datapath('E:/Data/GN/GoogleNews-vectors-negative300.bin'), binary=True)
embedding_vectors = get_weight_matrix2(wv_from_bin, tokenizer.word_index.items())

print('embedding_vectors.shape() =============================')
print(embedding_vectors.shape)

# create the embedding layer
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_length, trainable=False)

# fit network
embeding_dim = 300
filter_sizes = (1, 2, 3, 4)
num_filters = 100
dropout_prob = (0.0, 0.5)
batch_size = 64
num_epochs = 100
print('max_length', max_length)
input_shape = (max_length,)
model_input = Input(shape=input_shape)
zz = embedding_layer(model_input)


BUFFER_SIZE = len(Xtrain)
BATCH_SIZE = 64
steps_per_epoch = len(Xtrain)//BATCH_SIZE
embedding_dim = 256
units = 1024
#vocab_inp_size = len(inp_lang.word_index)+1
#vocab_tar_size = len(targ_lang.word_index)+1


embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
gru = tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')
x = embedding(Xtrain)
hidden = tf.zeros((BATCH_SIZE, units))
output, state = gru(x, initial_state=hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(x.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(hidden.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(hidden, x)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

