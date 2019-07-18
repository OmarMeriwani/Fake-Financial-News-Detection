import pandas as pd
from nltk.tokenize import sent_tokenize
import string
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from stanfordcorenlp import StanfordCoreNLP
import os

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)


df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/articles1.csv',header=0,encoding='ISO-8859-1')
df2 = pd.DataFrame(columns=['id','title','publication','author','date','year','month','url','content'])
seq = 0

for i in range(0,len(df)):
    #id,title,publication,author,date,year,month,url,content
    title= df.loc[i].values[2]
    print(title)
    lastDash = 0

    for j in range(0,len(title) - 1):
        c = title[len(title) - 1 - j]
        if c == '-':
            lastDash = len(title) - 1 - j
            break
    if lastDash != 0:
        title = title[:-(len(title) - (lastDash))]

    publication = df.loc[i].values[3]
    author = df.loc[i].values[4]
    ArticleDate = df.loc[i].values[5]
    Year = df.loc[i].values[6]
    Month = df.loc[i].values[7]
    URL = df.loc[i].values[8]
    text = df.loc[i].values[9]
    sentences = []
    try:
        title = title.encode('ascii', errors='ignore').decode("utf-8")
        print(title)
        text = text.encode('ascii', errors='ignore').decode("utf-8")
        print(str(text))
    except:
        continue
    title = str(title)
    NER = scnlp.ner(title)
    print(NER)
    print('--------------------------------------------------------------------------')
