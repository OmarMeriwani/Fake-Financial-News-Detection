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
from keras.utils import np_utils
from keras.layers import LSTM
from keras.layers import Bidirectional
import tensorflow as tf
from keras.callbacks import  EarlyStopping
from keras.layers import RepeatVector
import os
from stanfordcorenlp import StanfordCoreNLP

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

stemmer = PorterStemmer()

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
    df = pd.read_csv(filename,header=0)
    mode = 'sentence'
    data = pd.DataFrame(columns=['title','effect'])
    prev = ''
    seq = 0
    for i in range(0,len(df)):
        sentence = df.loc[i][3]
        company = str(df.loc[i][2]).lower()
        #print(company)
        #print(sentence)
        originalSentence = scnlp.word_tokenize(sentence)
        sentenceList = []
        #for word in originalSentence:
        #    if str(word).isupper() == False:
        #        sentenceList.append(str(word))
        #    else:
        #        sentenceList.append('NER')
        #sentence = ' '.join(sentenceList)
        NER = scnlp.ner(str(sentence))
        POS = scnlp.pos_tag(str(sentence).lower())


        #sentence = ' '.join([w for w, n in NER if n == 'O'])
        sentenceList = []
        for i in range(0,len(NER)):
            w = NER[i][0]
            n = NER[i][1]
            pos = NER[i][1]
            #print(w, n)

            #if pos == 'NNP':
            #    sentenceList.append('NER')
            #    continue
            #if str(w).lower() in scnlp.word_tokenize(company):
            #    sentenceList.append('NER')
            #    continue
            #if str(w).lower() == company:
            #    sentenceList.append('NER')
            #    continue
            if str(n) == 'O' :
                sentenceList.append(w)
            else:
                sentenceList.append('NER')
        sentence = ' '.join(sentenceList)
        effect = df.loc[i][4]
        #print(df.loc[i][4])
        if effect > 0:
            effect = 1
        else:
            effect = 0
        sentence = doc_to_clean_lines(sentence,vocab)
        if sentence.strip() != '':
            data.loc[seq] = [sentence,effect]
            print(sentence, effect)
            seq += 1
    return data

data = readfile('SSIX News headlines Gold Standard EN.csv')
headlines = data[['title']]
effects = data[['effect']]

x_train, x_test, y_train, y_test = train_test_split(headlines,effects,test_size=0.5)
traindata = np.array(x_train)
testdata = np.array(x_test)

y_test = np_utils.to_categorical(y_test,num_classes=2)
y_train = np_utils.to_categorical(y_train,num_classes=2)
print('y_train',y_test)
print('y_train',y_train)

train_docs = traindata[:,0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

# define training labels
test_docs = testdata[:,0]
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
# define model

# fit network
embeding_dim = 300
filter_sizes = (1,2,3,4)
num_filters = 100
dropout_prob = (0.0, 0.5)
batch_size = 64
num_epochs = 500
print('max_length',max_length)
input_shape = (max_length,)
model_input = Input(shape=input_shape)
zz = embedding_layer(model_input)

model_output = Dense(20, activation="sigmoid" , bias_initializer='zeros')(zz)
model_output = Dense(20)(model_output)
model_output = Dense(2)(model_output)
model_output = Dense(2, activation="selu")(model_output)
model = Model(model_input, model_output)
model.compile(loss="categorical_hinge", optimizer="adam", metrics=["accuracy"])
model.summary(85)


model.fit(Xtrain, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(Xtest, y_test), verbose=2)

loss, acc = model.evaluate(Xtest, y_test, verbose=2)
print('Test Accuracy: %f' % (acc*100))
