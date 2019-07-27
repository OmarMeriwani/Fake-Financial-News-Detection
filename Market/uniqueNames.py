import pandas as pd
import nltk
import csv
def removetype(name):
    """
    Inc., Group Inc., Ltd., Holdings Inc., Worldwide Holdings Inc., L.P., Class A, Worldwide, Holdings, Holding, Corporation, Ltd, Inc, Industries
    """
    name = str(name).replace('Inc.','')
    name = name.replace('Ltd.','')
    name = name.replace(' Ltd ','')
    name = name.replace('Worldwide Holdings','')
    name = name.replace('L.P.','')
    name = name.replace('Holding','')
    name = name.replace('Corporation','')
    name = name.replace('Sponsored ADR','')
    name = name.replace(' ETF','')
    return name

def getremoveclasses(name):
    tokens = nltk.tokenize.word_tokenize(name)
    for i in range(0, len(tokens)):
        try:
            if tokens[i] == 'Class':
                if tokens[i+1] in ['A','B','C','D','E','F','G','H','I']:
                    return ' '.join(tokens[:i]),'Class', tokens[i+1]
            if tokens[i] == 'Series':
                if str(tokens[i+1]) in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    return ' '.join(tokens[:i]),'Series', tokens[i+1]
        except:
            continue
df = pd.read_csv('symbols.csv')
namesList = []
previous = ''
count = 1
for i in range(0,len(df)):
    #,currency,date,exchange,iexId,isEnabled,name,region,symbol,type
    name = df.loc[i]['name']
    symbol = df.loc[i]['symbol']

    name = removetype(name)
    n = getremoveclasses(name)
    if n is not None:
        name = n[0]
    tokens = nltk.tokenize.word_tokenize(name)
    if len(tokens) >= 2:
        name = tokens[0]+ ' '+ tokens[1]
    else:
        name =tokens[0]
    if name != previous:
        previous = name
        count = 1
    else:
        count += 1

    namesList.append([name, symbol, count])
namesWithMoreThanOne = [i for i,t,u in namesList if u > 1]
namesList = [i for i in namesList if i[0] not in namesWithMoreThanOne ]
for n in namesList:
    print(n)
df2 = pd.DataFrame(namesList)
df2.to_csv('UniqueCompanyNames.csv')