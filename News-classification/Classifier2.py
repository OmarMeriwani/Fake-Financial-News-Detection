import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk.tokenize
def normalize_text(s):
    s = s.lower()
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)
    return s
alltokens = []
classifiedrows = 400000
df3 = pd.DataFrame(columns=['title'])
def get_vocabulary(doc,encoding,textIndex,encodeDecode):
    #'ISO-8859-1'
    df = pd.read_csv(doc, header=0, encoding=encoding)
    df = df[:classifiedrows]
    atokens = []
    for i in range(0,len(df)):
        sentence = df.loc[i][textIndex]
        sentence = normalize_text(sentence)
        if encodeDecode == True:
            sentence = sentence.encode('ascii', errors='ignore').decode("utf-8")
        df3.loc[i] = sentence
        tokens = nltk.tokenize.word_tokenize(sentence)
        for t in tokens:
            atokens.append(t)
    atokens = set(atokens)
    return atokens

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


#df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/articles1.csv',header=0,encoding='ISO-8859-1')
#alltokens = get_vocabulary('C:/Users/Omar/Documents/MSc Project/Datasets/articles1.csv','ISO-8859-1',2,True)
alltokens = get_vocabulary('C:/Users/Omar/Documents/MSc Project/Datasets/uci-news-aggregator.csv','utf-8',1,False)
df_store_vocab = pd.DataFrame(columns=['word'])
seq = 0
for i in alltokens:
    df_store_vocab.loc[seq] = i
    seq += 1
#df_store_vocab.to_csv('vocab.csv')
print(alltokens)
vectorizer = CountVectorizer(vocabulary=alltokens)
#print('DF3:',df3)
x2 = vectorizer.fit_transform(df3['title'])
print('SHAPE2: ',x2.shape)
news = pd.read_csv("C:/Users/Omar/Documents/MSc Project/Datasets/uci-news-aggregator.csv")
news = news[:classifiedrows]
seq = 0
df2 = pd.DataFrame(columns=['title','category'])
for i in range(0,len(news)):
    sentence = news.loc[i][1]
    sentence = normalize_text(sentence)
    #tokens = nltk.tokenize.word_tokenize(sentence)
    #tokens2 = []
    '''for t in tokens:
        if t in alltokens:
            tokens2.append(t)
    sentence = ' '.join(tokens2)'''
    #ID,TITLE,URL,PUBLISHER,CATEGORY,STORY,HOSTNAME,TIMESTAMP
    category = news.loc[i][4]
    r = [sentence, category]
    df2.loc[seq] = r
    seq += 1
print(df2)

#Vectorizer
x = vectorizer.fit_transform(df2['title'])
pickle.dump(vectorizer.vocabulary_, open('vocab.pkl', 'wb'))

print('SHAPE: ',x.shape)
#Change categories into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(df2['category'])
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

pickle.dump(mlp, open('MLPClassifier4.pkl', 'wb'))
score = mlp.score(x_test, y_test)
print(score)

y2 = mlp.predict(x2)
#for i in range(0,x2.shape[0]):
#    print(df3.loc[i][0],encoder.classes_[y2[i]])
print(encoder.classes_)
#b : business -- t : science and technology -- e : entertainment -- m : health
