import pandas as pd
import numpy as np
import re
from stanfordcorenlp import StanfordCoreNLP
import os
import sys
import nltk
#df_smbls = pd.read_csv('symbols.csv',header=0)
from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from keras.layers import Embedding
from numpy import zeros
from string import punctuation
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

df = pd.read_csv('BusinessTitlesFull.csv',header=0)
df2 = pd.DataFrame(columns=['ID','TITLE','URL','PUBLISHER','CATEGORY','HOSTNAME','TIMESTAMP'])

#df_corp_names = pd.read_csv('C:/Users/Omar/Documents/GitHub/Fake-Financial-News-Detection/Market/UniqueCompanyNames.csv')
df_corp_names = pd.read_csv('../Market/UniqueCompanyNames.csv')

corp_names = df_corp_names.values.tolist()
print(corp_names)

corp_names_freq = {}
for i  in corp_names:
    tokens = nltk.tokenize.word_tokenize(i[0])
    for t in tokens:
        if t not in corp_names_freq:
            corp_names_freq[str(t).lower()] = 1
        else:
            corp_names_freq[t] = corp_names_freq.get(str(t).lower()) + 1

df3 = pd.DataFrame(columns=['corporates','title','number'])
seq = 0
namesDataset = []
NoNeedForData = 1
if NoNeedForData == 0:
    for i in range(0, 100000):
        title= str(df.loc[i].values[1])
        foundCorporates = ''
        foundCorporatesList = []
        disable_single_words = 1
        #title = title.encode('ascii', errors='ignore').decode("utf-8")
        TitleTokens = []
        pos_title = scnlp.pos_tag(title)
        for p in pos_title:
            TitleTokens.append(str(p[0]).lower())
        TitleTokensCase = []
        for p in pos_title:
            TitleTokensCase.append( True if str(p[0])[0].isupper() else False)
        #print(title)
        #print(TitleTokens)
        points = re.findall(r'stock\ |stocks|share\ |shares', str(title).lower())
        if points != []:
            dollars = re.findall(r'\d+\.?\d+\%|[$|£|€|%]\d+\.?\d+\ ?[bln|billion|b\ |million|mln|m\ |k\ ]?|\d+\.?\d+\ ?point', title.lower())
            if dollars != []:
            #if numbers != []:
                for corp in corp_names:
                    #print('CORPP', corp)
                    tokens = nltk.tokenize.word_tokenize(corp[0])
                    firstOnly = tokens[0]
                    both = str(corp[0])
                    #print('BOTHH',both)
                    symbol = str(corp[1])
                    bothFound = 0
                    for j in range(0, len(TitleTokens)):
                        CurrentWord = str(TitleTokens[j]).lower()
                        NextWord = ''
                        CompanysFirstWord = str(tokens[0]).lower()
                        CompanysSecondWord = ''
                        try:
                            CompanysSecondWord = str(tokens[1]).lower()
                        except:
                            DONOTHING = 0
                        nextPOS = ''
                        try:
                            NextWord = str(TitleTokens[j + 1]).lower()
                            nextPOS = pos_title[j + 1][1]
                        except:
                            DONOTHING = 0
                        currentPOS = ''
                        try:
                            currentPOS = pos_title[j][1]
                        except Exception as e:
                            print(e)
                            print(pos_title)
                        '''if CurrentWord == 'ebay' and CompanysFirstWord.lower() == 'ebay':
                            print('CurrentWord == CompanysFirstWord', CurrentWord == CompanysFirstWord)
                            print('CompanysSecondWord == NULL', CompanysSecondWord == '')
                            print('NextWord == CompanysSecondWord', NextWord == CompanysSecondWord)
                            print('NextWord != NULL', NextWord != '')'''
                        if (CurrentWord == CompanysFirstWord and CompanysSecondWord == '' and TitleTokensCase[j] == True and 'NN' in currentPOS ) or  \
                                (CurrentWord == CompanysFirstWord and NextWord == CompanysSecondWord and NextWord != ''):
                            foundCorporates += ' | Corporate Name: '+ both +  ',' + str(j)
                            foundCorporatesList.append([str(j), both,'n'])
                            bothFound = 1
                        if disable_single_words == 0 and bothFound != 1 and CurrentWord == CompanysFirstWord and corp_names_freq.get(CompanysFirstWord) <= 5 \
                                and CompanysFirstWord not in stop_words  and ('NN' in nextPOS ) and 'NN' in currentPOS:
                            foundCorporates += ' | One Word:'+ both +  ',' + str(j)
                            foundCorporatesList.append([str(j), both,'n'])
                        if CurrentWord == symbol and len(symbol) > 3:
                            foundCorporates += ' | Corporate Symbol:'+ both + ',' + symbol + ',' + str(j)
                            foundCorporatesList.append([str(j), symbol,'s'])
                #print(points)
                if foundCorporates != '':
                    print('FOUND: ',[foundCorporatesList, title,  dollars])
                    seq += 1

                    df3.loc[seq] = [foundCorporatesList, title,  dollars]
    df3.to_csv('CompaniesWithNumbers.csv')

df4 = pd.read_csv('CompaniesWithNumbers.csv')
'''UPSDOWNS MODEL -------------------------------------------'''
def get_weight_matrix2(embedding, vocab):
    vocab_size2 = len(vocab) + 1
    weight_matrix = zeros((vocab_size2, 300))
    for word, i in vocab:
        vector = None
        try:
            vector = embedding.get_vector(word)
        except:
            continue
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix
data = []
for i in range(0,len(df4)):
    sentence = df4.loc[i][2]
    tokens = scnlp.word_tokenize(sentence)
    sentenceList = []
    for word in tokens:
        isAllUpperCase = True
        for letter in word:
            if letter.isupper() == False:
                isAllUpperCase = False
                break

        if isAllUpperCase == False:
            sentenceList.append(str(word))
        else:
            sentenceList.append('#ner')
    tokens = sentenceList
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if len(word) > 1]
    #tokens = [t for t in tokens if t not in [a for a,b,c in d[0]]]
    sentence = ' '.join(tokens)
    NER = scnlp.ner(str(sentence))
    POS = scnlp.pos_tag(str(sentence).lower())
    sentenceList = []
    for i in range(0, len(NER)):
        w = NER[i][0]
        n = NER[i][1]
        pos = NER[i][1]
        if str(w).isnumeric() == True:
            sentenceList.append('#num')
            continue
        if pos == 'NNP' and w != '#ner':
            sentenceList.append('#ner')
            continue
        if str(n) == 'O':
            sentenceList.append(w)
        else:
            sentenceList.append('#ner')
    sentence = ' '.join(sentenceList)
    if sentence.strip() != '':
        data.append([sentence])
data = np.array(data)
train_docs = data[:, 0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)
encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = 17
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
vocab_size = len(tokenizer.word_index) + 1
wv_from_bin = KeyedVectors.load_word2vec_format(datapath('E:/Data/GN/GoogleNews-vectors-negative300.bin'),
                                                binary=True)
embedding_vectors = get_weight_matrix2(wv_from_bin, tokenizer.word_index.items())
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_length, trainable=False)
model = load_model('../Sentiments/upsdowns_model.h5')
y = model.predict(Xtrain)
for i in range(0, len(y)):
    effect = ''
    if y[i][0] > y[i][1]:
        effect = 'DOWN'
    if y[i][0] < y[i][1]:
        effect = 'UP'
    print(df4.loc[i][2], effect)
'''-------------------------------------------UPSDOWNS MODEL '''
    #else:
    #    print('NOTFOUND: ',title)
#if firstOnly in title:
#    print(title, 'FIRST: ',firstOnly)
'''if both in title and len(tokens) > 1 :
    print(title, 'BOTH: ',both)
    break
if ' '+ symbol + ' ' in title and len(symbol) > 3:
    print(title, 'SYMBOL: ',symbol)
    break'''

'''
Single letters
Country names
High frequency words such as first, credit, new

ners = scnlp.ner(title)
for n in ners:
    if n[1] == 'ORGANIZATION':
        print(title, n[0])
        break
'''
