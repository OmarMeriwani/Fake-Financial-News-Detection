import nltk
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['Original_Name','Fixed_Name','UniqueID','SourceID',
                           'Type','Text','SentimentScore','EffectScore','SourceType','Text','Stemmed_Text',
                           'Lowercased','WithoutStopwords', 'WordNgrams','trigram','n4grams','n5grams','BoC',
                           'SenticentScores','Word2VecArrays','DateTime'])
PostsDF = pd.DataFrame(columns=['ID','Source','Type','Text','Effect','RelatedNER','Date','likes','views','Sentiment'])
WordsDf = pd.DataFrame(columns=['Post','Word','NER','POS','Stem','Word2Vec','Sentiment','trigram','n4grams','n5grams'])

gsdf = pd.read_csv('EnglishGS.csv')
texts = gsdf.values[:,1]
texts = texts.tolist()
tokens = []
for text in texts:
    regtokenzr = nltk.RegexpTokenizer('[A-Z,a-z,\$]\w+')
    tarray = regtokenzr.tokenize(str(text))
    tokens.append(tarray)
    print(text,':   ',tarray)
