import pandas as pd
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import nltk

sentence = 'Since 2010 he was failing to achieve anything good, but now he is achieving success'
tknzr = TweetTokenizer()
tokens = tknzr.tokenize(sentence)
postags = nltk.pos_tag(tokens)
print(postags)
#In the, On the, At, Since,
df = pd.read_csv('articles5k2.csv',header=0,encoding='ISO-8859-1')
for i in range(0,len(df)):
    title= df.loc[i].values[2]
    text = df.loc[i].values[9]
    sentences = []
    try:
        sentences = sent_tokenize(text)
    except:
        continue
    print(sentences[0],sentences[1])
    print('--------------------------------------------------------------------------')