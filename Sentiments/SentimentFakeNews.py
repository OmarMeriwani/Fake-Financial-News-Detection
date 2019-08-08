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
import keras.regularizers
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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

stemmer = PorterStemmer()

def get_weight_matrix2(embedding, vocab):
    vocab_size2 = len(vocab) + 1
    weight_matrix = zeros((vocab_size2, 300))
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
    data = pd.DataFrame(columns=['claim','label'])
    prev = ''
    seq = 0
    table = str.maketrans('', '', punctuation)

    for i in range(0,len(df)):
        sentence = df.loc[i][0]
        tokens = scnlp.word_tokenize(sentence)
        sentenceList = []
        for word in tokens:
            #print(word)
            isAllUpperCase = True
            for letter in word:
                if letter.isupper() == False:
                    isAllUpperCase = False
                    #print(word,letter, isAllUpperCase)
                    break

            if isAllUpperCase == False:
                sentenceList.append(str(word))
            else:
                sentenceList.append('#ner')
        tokens = sentenceList

        tokens = [w.translate(table) for w in tokens]
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if len(word) > 1]
        sentence = ' '.join(tokens)
        NER = scnlp.ner(str(sentence))
        POS = scnlp.pos_tag(str(sentence).lower())
        sentenceList = []
        for i in range(0,len(NER)):
            w = NER[i][0]
            n = NER[i][1]
            pos = NER[i][1]
            #print(w, n)
            if str(w).isnumeric() == True:
                sentenceList.append('#num')
                continue
            if pos == 'NNP' and w != '#ner':
                sentenceList.append('#ner')
                continue
            if str(n) == 'O' :
                sentenceList.append(w)
            else:
                sentenceList.append('#ner')
        sentence = ' '.join(sentenceList)
        effect = df.loc[i][1]
        if str(effect).lower() == 'true':
            effect = 1
        else:
            effect = 0
        if sentence.strip() != '':
            data.loc[seq] = [sentence,effect]
            print(sentence, effect)
            seq += 1
    return data

data = readfile('FakeNewsSA.csv')
headlines = data[['claim']]
labels = data[['label']]

x_train, x_test, y_train, y_test = train_test_split(headlines,labels,test_size=0.2)
traindata = np.array(x_train)
testdata = np.array(x_test)

y_testold = y_test
y_test = np_utils.to_categorical(y_test,num_classes=2)
print(y_testold, y_test)
y_train = np_utils.to_categorical(y_train,num_classes=2)
#print('y_train',y_test)
#print('y_train',y_train)

train_docs = traindata[:,0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

# define training labels
test_docs = testdata[:,0]
'''==============================================='''
# pad sequences
encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = max([len(s.split()) for s in train_docs])
print('max_length', max_length)
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

conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(zz)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks if len(conv_blocks) > 1 else conv_blocks[0])
z = Dropout(0.8)(z)
model_output = Dense(10, activation="sigmoid" , bias_initializer='zeros')(z)
model_output = Dense(10)(model_output)
model_output = Dropout(0.8)(model_output)
#model_output = Dense(2)(model_output)
model_output = Dense(2, activation="selu")(model_output)
model = Model(model_input, model_output)
max = 0
for i in range(100):
    model.compile(loss="categorical_hinge", optimizer="adam", metrics=["accuracy"])
    model.summary(85)
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience=50)
    history = model.fit(Xtrain, y_train, batch_size=batch_size, epochs=50,
              validation_data=(Xtest, y_test), verbose=2)
    print('History', history.history)
    #model.fit(Xtrain, ytrain, epochs=10, verbose=2)
    # evaluate
    loss, acc = model.evaluate(Xtest, y_test, verbose=2)
    print('Test Accuracy: %f' % (acc*100))
    if acc > max:
        max = acc
        model.save('sentimentAnalysis.h5')
