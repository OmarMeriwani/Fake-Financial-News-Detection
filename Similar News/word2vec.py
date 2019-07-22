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
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w.lower() in vocab]
		clean_lines = ' '.join(tokens)
	return clean_lines
def process_docs2(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines

# load the vocabulary
vocab_filename = 'vocabulary.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
vocab = [v.lower() for v in vocab]
def load_embedding2(binfile):
	# load embedding into memory, skip first line
	# create a map of words to vectors
	embedding = dict()
	for line in binfile:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix
def get_weight_matrix2(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 300))
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
	df = pd.read_csv(filename,header=0,sep='\t')
	mode = 'sentence' #all sentences or only full reviews (sentence,full)
	data = []
	prev = ''
	for i in range(0,len(df)):
		if mode == 'sentence':
			if prev != str(df.loc[i][1]):
				sentence = df.loc[i][2]
				prev = str(df.loc[i][1])
			else:
				continue
		else:
			sentence = df.loc[i][2]
		reviewPolarity = int(df.loc[i][3])
		'''tokens = []
		tknzr = RegexpTokenizer(r'\w+')
		t = tknzr.tokenize(sentence)
		for tk in t:
			tokens.append(tk)
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]'''
		sentence = doc_to_clean_lines(sentence,vocab)
		data.append([sentence,reviewPolarity])
	return data
def split(docs, percentage):
	length = len(docs)
	firstlength = int (length * percentage)
	training = docs[:firstlength]
	test = docs[firstlength:length]
	return training,test
data = np.array(readfile('train.csv'))
print(data.shape)
#print(data[:,0])
traindata, testdata = split(data,0.7)
print(testdata.shape)
print(traindata.shape)
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
#raw_embedding = load_embedding('embedding2.txt')
#raw_embedding = load_embedding('G:/Data/GN/GoogleNews-vectors-negative300.bin', binary=True)
wv_from_bin = KeyedVectors.load_word2vec_format(datapath('G:/Data/GN/GoogleNews-vectors-negative300.bin'), binary=True)
#raw_embedding = load_embedding(wv_from_bin)
# get vectors in the right order
#embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)

embedding_vectors = get_weight_matrix2(wv_from_bin, tokenizer.word_index.items())

print('embedding_vectors.shape() =============================')
print(embedding_vectors.shape)

# create the embedding layer
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_length, trainable=False)
# define model
'''model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])'''
model = Sequential()
model.add(embedding_layer)
model.add(Dense(128, activation='relu', input_dim=200))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#print(model.summary())

# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
#model = LogisticRegression(C=0.2, dual=True)
#model.fit(Xtrain, ytrain)

# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
#result = model.score(Xtest,ytest)
print('Test Accuracy: %f' % (acc*100))
#print(result)