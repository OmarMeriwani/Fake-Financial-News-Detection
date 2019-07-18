import pandas as pd
from nltk.tokenize import sent_tokenize
import string
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from stanfordcorenlp import StanfordCoreNLP
import os
import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer
sys.path.append("./Citations/verbsynonyms.py")
from nltk.stem.porter import *
from verbsynonyms import getVerbs

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

tokenizer = RegexpTokenizer('\s+|\:|\.', gaps=True)
MWETokenizer = MWETokenizer()
df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/Using Resources Dataset.csv',header=0)
#df2 = pd.DataFrame(columns=['id','reference','IsReferenced'])
seq = 0
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

for i in range(0,len(df)):
    #id,title,publication,author,date,year,month,url,content
    title= str(df.loc[i].values[0])
    title = title.replace('...','')
    lemTags = scnlp.pos_tag(title)
    isreferenced = df.loc[i].values[2]
    colonAvailable = 1 if (title.find(':') != -1) else 0
    tags = scnlp.pos_tag(title)
    tagsarr = []
    sayverbs = getVerbs('say')
    isSayVerb = 0
    nnpfound = 0
    nnp_followed_by_colon = 0
    mid = int((len(tags) -1) / 2)
    for t in lemTags:
        if 'V' in t[1]:
            for j in sayverbs:
                if stemmer.stem(str(t[0]).lower()) == str(j).lower():
                    isSayVerb = 1
                    print('SayVerb',t[0])
                    break
    for i in range(0,mid):
        word = tags[i][1]
        if nnpfound == 1 and word == ':':
            nnp_followed_by_colon = 1
            break
        if word == 'NNP':
            nnpfound = 1
        else:
            nnpfound = 0
    nnp_preceeded_by_colon = 0
    for i in range(0,mid ):
        word = tags[len(tags) -1 - i][1]
        word2 = tags[len(tags) -1 - i][0]

        #print(word2)
        if  word == 'NNP':
            nnpfound = 1
        if nnpfound == 1 and word == ':':
            nnp_preceeded_by_colon = 1
            break
        if word != 'NNP':
            nnpfound = 0
    #if colonAvailable == 1 :
        #print(tags)
        #ORGANIZATION, COUNTRY, PERSON
    print(title)
    print( 'isreferenced', isreferenced,'colonAvailable', colonAvailable, 'nnp_followed_by_colon:', nnp_followed_by_colon,'nnp_preceeded_by_colon',nnp_preceeded_by_colon)
    print('--------------------------------------------------------------------------')
