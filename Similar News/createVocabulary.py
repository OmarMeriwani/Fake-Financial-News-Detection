import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def readfile(filename):
    df = pd.read_csv(filename,header=0)
    mode = 's'
    data = ''
    prev = ''
    for i in range(0,len(df)):
        sentence = df.loc[i][1]
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
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if len(word) > 1]
    return tokens

data = readfile('NewsGroups1300.csv')
tokens = clean_doc(data)
vocabulary = Counter()
vocabulary.update(tokens)
items = [word for word, count in vocabulary.items()]
items = '\n'.join(items)
file = open('vocabulary.txt', 'w')
file.write(items)
file.close()

