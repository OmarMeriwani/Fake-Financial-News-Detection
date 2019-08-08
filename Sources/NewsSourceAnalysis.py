import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from stanfordcorenlp import StanfordCoreNLP
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import os
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense

df = pd.read_csv('Resources.csv',header=0)

data = []
labels = []
prev = ''
for i in range(0,len(df)):
    #claim,label,date,sources
    sources =  str(df.loc[i][1]).replace('www','').replace('.','').split(';')
    sources2 = []
    for s in sources:
        if s == '':
            continue
        else:
            sources2.append(s)
    if sources2 == []:
        continue
    sources = ' '.join(sources2)
    #print(sources)
    label = df.loc[i][0]
    if str(label).lower()== 'true':
        label = 1
    else:
        label = 0
    data.append(sources)
    labels.append(label)
x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2)

tfidf = TfidfVectorizer(analyzer='word')
_ = tfidf.fit(x_train)
train_tfidf = tfidf.transform(x_train)
test_tfidf = tfidf.transform(x_test)
print ('Transforming final test dataset')

model1 = MLPClassifier(hidden_layer_sizes = (20,20,20), solver ='lbfgs', random_state=50)
model1.fit(train_tfidf,y_train)
#print(model1.score(test_tfidf,y_test))

model = Sequential()
model.add(Dense(18, input_dim=(train_tfidf.shape[1] )))
model.add(Dense(20))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='hinge', metrics=['accuracy'])
model.fit(train_tfidf, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(test_tfidf, y_test))
score = model.evaluate(test_tfidf, y_test, batch_size=128, verbose=2)
print ('SCORE: ',score[1])

'''
Adam: 63%
Adam with two layers: 68%
lbfgs: 75%

KERAS
'''
