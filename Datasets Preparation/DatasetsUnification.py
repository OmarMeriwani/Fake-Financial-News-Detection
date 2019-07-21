import pandas as pd
import re
import sys
from urllib.parse import urlparse
from dateutil.parser import parse
import pickle
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0, '../News-classification/TF-With-2ND-Dataset.py')
from 'TF-With-2ND-Dataset' import normalize_text

df_main = pd.DataFrame(columns=['claim', 'type', 'label', 'date','sources'])
df_snopes = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/rumor-citation/snopes.csv')
'''
0. snopes_page, 1. topic, 2. claim, 3. claim_label, 4. date_published, 5. date_updated, 6. page_url,page_is_example,page_is_image_credit,page_is_archived,page_is_first_citation,tags
topic (business), claim, claim_label (FALSE, TRUE, mfalse, mtrue), date_published, page_url
'''
seq = 0
previous = ''
sources = ''
'''
for i in range(0, len(df_snopes)):
    claim = str(df_snopes.loc[i].values[2])
    claim = claim.replace('            See Example(s)','')
    topic = df_snopes.loc[i].values[1]
    if topic != 'business':
        continue

    date = df_snopes.loc[i].values[4]
    dt = parse(date)
    #print('DATE: ',date, dt.date())
    label = df_snopes.loc[i].values[3]
    if label == 'mfalse':
        label = 'false'
    if label == 'mtrue':
        label = 'true'
    parsed_uri = urlparse(str(df_snopes.loc[i].values[6]) )
    result = '{uri.netloc}'.format(uri=parsed_uri)
    result = str(result)
    result.replace('www.','')
    sources += str(result) + ';'

    if claim != previous:

        df_main.loc[seq] = [claim, topic, label, str(dt.date()) , sources]
        print([claim, topic, label, str(dt.date()), sources])
        sources = ''
        previous = claim
        seq += 1
print(df_main)
'''
'''
emergent.csv
0.emergent_page, 1.claim,2.claim_description,3.claim_label,4.tags,5.claim_source_domain,
6.claim_course_url,7.date,8.body,9.page_domain,10.page_url,11.page_headline,12.page_position,13.page_shares,14.page_order
1.claim, 3.claim_label (FALSE, TRUE), 5.claim_source_domain, 7.date
'''
def getReadyVocabulary():
    atokens = []
    dfVocab = pd.read_csv('../News-classification/vocab.csv')
    for i in range(0,len(dfVocab)):
        sentence = dfVocab.loc[i][1]
        #print(sentence)
        atokens.append(sentence)
    atokens = set(atokens)
    return atokens
print('Getting vocabulary...')
vocab = getReadyVocabulary()
print(vocab)
vec = CountVectorizer(vocabulary=pickle.load(open('../News-classification/vocab.pkl')))

print('Loading model')
file = open(r"../News-classification/MLPClassifier4.pkl", 'rb')
mlp = pickle.load(file)

print('Getting emergent.csv dataset...')
df_emergent = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/rumor-citation/emergent.csv')
df_emergent_pred = pd.DataFrame(columns=['claim'])
seq = 0
for i in range(0, len(df_emergent)):
    claim = str(df_emergent.loc[i].values[1])
    claim = normalize_text(claim)
    df_emergent_pred.loc[seq] = [claim]

x2 = vec.fit_transform(df_emergent_pred['claim'])
y2 = mlp.predict(x2)
print(y2)
for i in range(0, len(df_emergent)):
    #1.claim, 3.claim_label (FALSE, TRUE), 5.claim_source_domain, 7.date
    claim = str(df_emergent.loc[i].values[1]).lower()
    claim = claim.replace('            See Example(s)','')
    topic = 0
    date = df_emergent.loc[i].values[7]
    dt = parse(date)
    #print('DATE: ',date, dt.date())
    label = df_emergent.loc[i].values[3]
    parsed_uri = urlparse(str(df_snopes.loc[i].values[5]) )
    result = '{uri.netloc}'.format(uri=parsed_uri)
    result = str(result)
    result.replace('www.','')
    sources += str(result) + ';'
    if claim != previous:

        df_main.loc[seq] = [claim, topic, label, str(dt.date()) , sources]
        print([claim, topic, label, str(dt.date()), sources])
        sources = ''
        previous = claim
        seq += 1
x2 = vec.fit_transform(df_main['claim'])
y2 = mlp.predict(x2)
for i in range(0,x2.shape[0]):
    print(df_main.loc[i][0],y2[i])

'''
politifact.csv
politifact_page,claim,claim_source,claim_citation,claim_label,date_published,researched_by,edited_by,tags,page_citation,page_url,page_is_first_citation
claim, claim_label (barely_true, half-true, mostly-true, pants-fire, FALSE, TRUE), date_published, page_url
'''
