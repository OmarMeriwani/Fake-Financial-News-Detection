import pandas as pd
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
import spacy
import pysyntime
from pysyntime import SynTime
from dateutil import parser

#nlp = spacy.load('D://en_core_web_lg-2.0.0//en_core_web_lg-2.0.0//en_core_web_lg//en_core_web_lg-2.0.0')
spacy.load('en_core_web_sm')
sentence = 'Since 2010 he was failing to achieve anything good, but now he is achieving success'
tknzr = TweetTokenizer()
tokens = tknzr.tokenize(sentence)
synTime = SynTime()
psnt = pysyntime.TimeSegment()

text = 'The last 6 months surviving member of the team which first conquered Everest in 6 a.m. 17 Jan 1953 has died in a Derbyshire nursing home.'
date = '2016-10-10'
timeMLText = synTime.extractTimexFromText(text, date)
print(timeMLText)


df = pd.read_csv('articles5k2.csv',header=0,encoding='ISO-8859-1')
for i in range(0,len(df)):

    title= df.loc[i].values[2]
    text = df.loc[i].values[9]
    sentences = []
    try:
        datee = parser.parse(df.loc[i].values[5])
        datee = datee.strftime('%Y-%m-%d')
        sentences = sent_tokenize(text)
        print(datee, synTime.extractTimexFromText(sentences[0] + ' ' + sentences[1], datee))

    except:
        continue

    print('--------------------------------------------------------------------------')
