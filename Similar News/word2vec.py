import numpy as np
from string import punctuation
from os import listdir
import pandas as pd
from numpy import zeros
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn.model_selection import train_test_split
from nltk.stem.porter import *
from createVocabulary import clean_doc

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
        #print('Line Before: ', line)
        clean_lines = ' '.join(clean_doc(line))
        #print('Line After: ', clean_lines)

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
    data = []
    prev = ''
    for i in range(0,len(df)):
        sentence = df.loc[i][1]
        group = int(df.loc[i][2])
        timestamp = df.loc[i][3]
        sentence = doc_to_clean_lines(sentence,vocab)
        data.append([sentence,timestamp,group])
    return data

def sortByGroup(val):
    return val[2]

def split(docs, percentage):
    docs.sort(key=sortByGroup)
    length = len(docs)
    groups = []
    test = []
    training = []
    previousGroup = 0
    for i in docs:
        if i[2] != previousGroup and previousGroup == 0:
            previousGroup = i[2]
            groups.append([i[0],i[2]])
        if i[2] == previousGroup and previousGroup != 0:
            groups.append([i[0],i[2]])
        if i[2] != previousGroup and previousGroup != 0:
            gLength = groups.__len__()
            testsize = int(gLength * percentage)
            groupsTraining, groupsTest  = train_test_split(groups,test_size=percentage)
            for t in groupsTraining:
                training.append(t)
            for t in groupsTest:
                test.append(t)
            #print('groupsTest', [g for g in groupsTest])
            #print('groupsTraining', [g for g in groupsTraining])
            groups = []
            groups.append([i[0], i[2]])
            previousGroup = i[2]
            print(i[2])

    #print('GROUPS ',groups)
    firstlength = int (length * percentage)
    #training = docs[:firstlength]
    #test = docs[firstlength:length]
    return training,test


data = readfile('NewsGroups1300.csv')
#print(data.shape)
traindata, testdata = split(data,0.2)
traindata = np.array(traindata)
testdata = np.array(testdata)
#traindata, testdata = train_test_split(data[1],data[2],test_size=0.33,stratify=data[2])
#print(testdata.shape)
#print(traindata.shape)
train_docs = traindata[:,0]
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)

'''==============================================='''
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = traindata[:,1]
'''==============================================='''
# create the tokenizer

# load all test reviews
test_docs = testdata[:,0]
'''==============================================='''
# pad sequences
encoded_docs = tokenizer.texts_to_sequences(test_docs)

Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = testdata[:,1]

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
model = Sequential()
model.add(embedding_layer)
model.add(Dense(128, activation='relu', input_dim=200))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())
#Test Accuracy: 0.692042
#
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
