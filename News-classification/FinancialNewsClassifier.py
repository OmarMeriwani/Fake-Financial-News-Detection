import re
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

news = pd.read_csv("C:/Users/Omar/Documents/MSc Project/Datasets/uci-news-aggregator.csv")
print(news.head())
news = news[:4000]

def normalize_text(s):
    s = s.lower()
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)
    return s

news['TEXT'] = [normalize_text(s) for s in news['TITLE']]

#Vectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(news['TEXT'])
#Change categories into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#nb = MultinomialNB()
#nb.fit(x_train, y_train)
#score  = nb.score(x_test, y_test)

#Use ANN
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(20,20,20))
mlp.fit(x_train,y_train)

#pickle.dump(mlp, open('MLPClassifier20.sav', 'wb'))
score = mlp.score(x_test, y_test)
print(score)