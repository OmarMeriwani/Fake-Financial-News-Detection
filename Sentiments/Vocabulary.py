import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP
import os
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)


def readfile(filename):
    #id,Company Name (Original),Company Name (Fixed),Text,sentiment score,# Scores
    df = pd.read_csv(filename,header=0)
    mode = 's'
    data = ''
    prev = ''
    for i in range(0,len(df)):
        sentence = df.loc[i][3]
        data = data + '.' + sentence
    return data


def clean_doc(doc):
    doc = doc.encode('ascii', errors='ignore').decode("utf-8")
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    #tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if len(word) > 1 and str(word).isalpha() == True ]
    return tokens
def calc():
    data = readfile('SSIX News headlines Gold Standard EN.csv')
    tokens = clean_doc(data)
    vocabulary = Counter()
    vocabulary.update(tokens)
    items = [word for word, count in vocabulary.items()]
    items = '\n'.join(items)
    file = open('vocabulary.txt', 'w')
    file.write(items)
    file.close()

