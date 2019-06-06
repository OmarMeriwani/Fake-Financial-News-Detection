import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk.tokenize
import re, string, timeit

df = pd.read_csv('EnglishGS.csv',header=0)
'''Unique,Text,Average Score,Type,source_id'''
#print(df['Text'])

dfResult = pd.DataFrame(columns=['Unique','Text','Average Score','Type','source_id','Tokens','LN3','LN4','LN5'])
for i in range(0,len(df)):

    text = df.loc[i].values[1]
    print('BEFORE',text)

    text = str(text).replace('\n','')
    text = str(text).replace('\\n','')
    '''Remove NER'''

    newSentence = str(text)
    ns = []

    #newSentence = ' '.join([i for i in str(text).split(' ') if i[0] != '$' and len(i) > ])
    for i in newSentence.split(' '):

        if len(i) <= 1:
            ns.append(i)
            continue
        if i[0] != '$':
            ns.append(i)
            continue
        else:
            if (i[1].isalpha() == True):
                ns.append('#NER')
            else:
                #print(i)
                ns.append(i)
    newSentence = ' '.join(ns)
    #newSentence = re.sub(r'\$[A-Z][A-Z][A-Z][A-Z] ', " ", newSentence)
    newSentence = re.sub(" [A-Z]{2}[A-Z]+ ", " ", newSentence)

    '''Lower case'''
    newSentence = ' '.join( [t.strip().lower()  for t in newSentence.split(' ') ])

    '''Remove double spaces'''
    newSentence = newSentence.replace('  ','')
    '''Remove time'''
    newSentence = re.sub(r'\d+\:\d+ (p|a)m', '', newSentence)
    '''Remove links'''
    newSentence = re.sub(r'http\S+', '', newSentence)
    '''Remove special characters'''
    newSentence = newSentence.translate(str.maketrans('', '', string.punctuation))
    '''Remove numbers'''
    #newSentence = re.sub("\d+\$|\d+", " ", newSentence)
    '''Remove multiple spaces'''
    newSentence = re.sub(' +', ' ',newSentence)
    print('AFTER',newSentence)

    '''Tokenization'''