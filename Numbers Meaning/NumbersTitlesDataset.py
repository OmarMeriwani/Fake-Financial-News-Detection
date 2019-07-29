import pandas as pd
import re
from stanfordcorenlp import StanfordCoreNLP
import os
import sys
import nltk
#df_smbls = pd.read_csv('symbols.csv',header=0)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/BusinessTitlesFull.csv',header=0)
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
for i in range(0, 100000):
    title= str(df.loc[i].values[1])
    foundCorporates = ''
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
        dollars = re.findall(r'\d+\.?\d+\%|[$|£|€|%]\d+\.?\d+\ ?[bln|billion|b\ |million|mln|m\ |k\ ]?', title.lower())
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
                        bothFound = 1
                    if disable_single_words == 0 and bothFound != 1 and CurrentWord == CompanysFirstWord and corp_names_freq.get(CompanysFirstWord) <= 5 \
                            and CompanysFirstWord not in stop_words  and ('NN' in nextPOS ) and 'NN' in currentPOS:
                        foundCorporates += ' | One Word:'+ both +  ',' + str(j)
                    if CurrentWord == symbol and len(symbol) > 3:
                        foundCorporates += ' | Corporate Symbol:'+ both + ',' + symbol + ',' + str(j)
            #print(points)
            if foundCorporates != '':
                print('FOUND: ',title, foundCorporates, points, dollars)
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
