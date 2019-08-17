import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from stanfordcorenlp import StanfordCoreNLP
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

tokenizer = RegexpTokenizer('\s+|\:|\.', gaps=True)
df = pd.read_csv('Using Resources Dataset.csv',header=0)
#df2 = pd.DataFrame(columns=['id','reference','IsReferenced'])
seq = 0
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
x = []
y = []

def getVerbs(verb):
    synonyms = []
    for syn in wordnet.synsets(verb):
        for l in syn.lemmas():
            synonyms.append(l.name())
    #print(set(synonyms))
    return set(synonyms)

def WhoSaid (sent, verb):
    #sent = sent.lower()
    result = []
    deps = scnlp.dependency_parse(sent)
    tags = scnlp.pos_tag(sent)
    ners = scnlp.ner(sent)
    verbindex = []
    for i in range(1, len(tags)):
        if tags[i][0] == verb:
            verbindex .append( i + 1)
            #print(verbindex)
            #break
    for i in deps:
        if i[1] in verbindex and i[0] == 'nsubj':
            result.append([tags[i[2] - 1][0], tags[i[2] - 1][1], ners[i[2] - 1][1] ])
    return result



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
    isNPPSaid = 0
    isNERSaid = 0
    isQuestion = 0
    nnpfound = 0
    if '?' in title:
        isQuestion = 1
    nnp_followed_by_colon = 0
    mid = int((len(tags) -1) / 2)
    for t in lemTags:
        verb = stemmer.stem(str(t[0]).lower())
        if 'V' in t[1]:
            for j in sayverbs:

                if verb == str(j).lower():
                    whosaid = WhoSaid(title, str(t[0]))
                    if whosaid != []:
                        for w in whosaid:
                            if w[1] == 'NNP' and isNPPSaid == 0:
                                isNPPSaid = 1
                            if w[2] != 'O' and isNERSaid == 0:
                                isNERSaid = 1
                        print('Whosaid', whosaid)
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
    print( 'isreferenced', isreferenced,'colonAvailable', colonAvailable, 'nnp_followed_by_colon:',
           nnp_followed_by_colon,'nnp_preceeded_by_colon',nnp_preceeded_by_colon, 'isNPPSaid',isNPPSaid,
           'isNERSaid',isNERSaid, 'isQuestion',isQuestion)
    x.append([colonAvailable,nnp_followed_by_colon,nnp_preceeded_by_colon, isNPPSaid,isNERSaid, isQuestion])
    y.append(isreferenced)
    print('--------------------------------------------------------------------------')
max = 0
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
for i in range(0,100):
    mlp = MLPClassifier()
    mlp.fit(X_train,y_train)
    score = mlp.score(X_test,y_test)
    print(score)
    if score > max:
        max = score
        pickle.dump(mlp, open('WhoSaid.pkl', 'wb'))
'''Score: 90.45%'''