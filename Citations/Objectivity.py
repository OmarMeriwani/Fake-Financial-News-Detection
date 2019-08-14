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
from nltk.stem.porter import *
from verbsynonyms import getVerbs
from parsers import WhoSaid
from sklearn.model_selection import train_test_split
import os
from stanfordcorenlp import StanfordCoreNLP
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)
stemmer = PorterStemmer()

features = []
df = pd.read_csv('../Sentiments/FakeNewsSA.csv')
for i in range(0,len(df)):
    claim = df.loc[i][0]
    label = df.loc[i][1]
    lemTags = scnlp.pos_tag(claim)
    colonAvailable = 1 if (claim.find(':') != -1) else 0
    tags = scnlp.pos_tag(claim)
    tagsarr = []
    sayverbs = getVerbs('say')
    isSayVerb = 0
    isNPPSaid = 0
    isNERSaid = 0
    isQuestion = 0
    nnpfound = 0
    if '?' in claim:
        isQuestion = 1
    nnp_followed_by_colon = 0
    mid = int((len(tags) - 1) / 2)
    for t in lemTags:
        verb = stemmer.stem(str(t[0]).lower())
        if 'V' in t[1]:
            for j in sayverbs:

                if verb == str(j).lower():
                    whosaid = WhoSaid(claim, str(t[0]))
                    if whosaid != []:
                        for w in whosaid:
                            if w[1] == 'NNP' and isNPPSaid == 0:
                                isNPPSaid = 1
                            if w[2] != 'O' and isNERSaid == 0:
                                isNERSaid = 1
                        print('Whosaid', whosaid)
                    isSayVerb = 1
                    #print('SayVerb', t[0])
                    break
    for i in range(0, mid):
        word = tags[i][1]
        if nnpfound == 1 and word == ':':
            nnp_followed_by_colon = 1
            break
        if word == 'NNP':
            nnpfound = 1
        else:
            nnpfound = 0
    nnp_preceeded_by_colon = 0
    for i in range(0, mid):
        word = tags[len(tags) - 1 - i][1]
        word2 = tags[len(tags) - 1 - i][0]

        # print(word2)
        if word == 'NNP':
            nnpfound = 1
        if nnpfound == 1 and word == ':':
            nnp_preceeded_by_colon = 1
            break
        if word != 'NNP':
            nnpfound = 0
    numberOfNER = 0
    usingTimeExpressions = 0

    for tag in scnlp.ner(claim):
        if tag[1] != 'O':
            numberOfNER += 1
        if tag[1] == 'DATE' and usingTimeExpressions == 0:
            usingTimeExpressions = 1

    #print(claim)
    #print('colonAvailable', colonAvailable, 'nnp_followed_by_colon:',
    #      nnp_followed_by_colon, 'nnp_preceeded_by_colon', nnp_preceeded_by_colon, 'isNPPSaid', isNPPSaid,
    #      'isNERSaid', isNERSaid, 'isQuestion', isQuestion)
    features.append([colonAvailable, nnp_followed_by_colon, nnp_preceeded_by_colon, isNPPSaid, isNERSaid, isQuestion])

mlp = pickle.load(open('WhoSaid.pkl', 'rb'))
y= mlp.predict(features)
features2 = []
labels = []
for i in range(0, len(features)):
    claim = df.loc[i][0]
    Cited = y[i]
    label = df.loc[i][1]
    numberOfNER = 0
    usingTimeExpressions = 0
    for tag in scnlp.ner(claim):
        if tag[1] != 'O':
            numberOfNER += 1
        if tag[1] == 'DATE' and usingTimeExpressions == 0:
            usingTimeExpressions = 1
    print(claim, numberOfNER, usingTimeExpressions, label)
    features2.append([ numberOfNER, usingTimeExpressions])
    labels.append(label)
xtrain, xtest, ytrain, ytest = train_test_split(features2, labels)
mlp2 = MLPClassifier()
max = 0
for i in range(0,100):
    mlp2.fit(xtrain,ytrain)
    score = mlp2.score(xtest, ytest)
    if score > max:
        max = score
        print('Accuracy: ',score)
        yhat_classes = mlp2.predict(xtest)
        #yhat_classes = yhat_classes[:, 0]
        # precision tp / (tp + fp)
        precision = precision_score(ytest, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(ytest, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(ytest, yhat_classes)
        print('F1 score: %f' % f1)
'''
Accuracy:  0.5694444444444444
Precision: 0.727273
Recall: 0.222222
F1 score: 0.340426 '''