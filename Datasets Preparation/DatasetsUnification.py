import pandas as pd
import re
import sys
from urllib.parse import urlparse
from dateutil.parser import parse
import pickle
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0, 'C:/Users/Omar/Documents/GitHub/Fake-Financial-News-Detection/')
from preprocessing import normalize_text
print('Started data unification...')

df_main = pd.DataFrame(columns=['claim', 'type', 'label', 'date','sources'])
print('Reading snopes dataset...')
df_snopes = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/rumor-citation/snopes.csv')
'''
0. snopes_page, 1. topic, 2. claim, 3. claim_label, 4. date_published, 5. date_updated, 6. page_url,page_is_example,page_is_image_credit,page_is_archived,page_is_first_citation,tags
topic (business), claim, claim_label (FALSE, TRUE, mfalse, mtrue), date_published, page_url
'''
seq = 0
mainseq = 0
previous = ''
sources = ''

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
    if label == 'Unverified' or label == 'mixture':
        continue
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

        df_main.loc[mainseq] = [claim, topic, label, str(dt.date()) , sources]
        print([claim, topic, label, str(dt.date()), sources])
        sources = ''
        previous = claim
        mainseq += 1
print(df_main)

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
vocab = getReadyVocabulary()

print(vocab)'''
print('Getting vocabulary...')
vec = CountVectorizer(vocabulary=pickle.load(open('../News-classification/vocab.pkl', 'rb')))

print('Loading model')
file = open(r"../News-classification/MLPClassifier4.pkl", 'rb')
mlp = pickle.load(file)
'''
emergent.csv
0.emergent_page, 1.claim,2.claim_description,3.claim_label,4.tags,5.claim_source_domain,
6.claim_course_url,7.date,8.body,9.page_domain,10.page_url,11.page_headline,12.page_position,13.page_shares,14.page_order
1.claim, 3.claim_label (FALSE, TRUE), 5.claim_source_domain, 7.date
'''

print('Getting emergent.csv dataset...')
df_emergent = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/rumor-citation/emergent.csv')
df_emergent_pred = pd.DataFrame(columns=['claim'])
seq = 0
for i in range(0, len(df_emergent)):
    claim = str(df_emergent.loc[i].values[1])
    claim = normalize_text(claim)
    claim = claim.replace('claim: ','')
    df_emergent_pred.loc[seq] = [claim]
    seq += 1

print(len(df_emergent_pred))
print('Fitting countvectorizer...')
x2 = vec.fit_transform(df_emergent_pred['claim'])
print('Prediction...')
y2 = mlp.predict(x2)
print('Predictions:')
seq = 0
previous = ''
sources = ''
for i in range(0, len(df_emergent)):
    #1.claim, 3.claim_label (FALSE, TRUE), 5.claim_source_domain, 7.date
    claim = str(df_emergent.loc[i].values[1]).replace('Claim: ','')
    topic = y2[i]
    if topic != 0:
        continue
    date = df_emergent.loc[i].values[7]
    dt = parse(date)
    #print('DATE: ',date, dt.date())
    label = df_emergent.loc[i].values[3]
    parsed_uri = urlparse(str(df_emergent.loc[i].values[10]) )
    result = '{uri.netloc}'.format(uri=parsed_uri)
    result = str(result)
    result.replace('www.','')
    sources += str(result) + ';'
    if claim != previous:
        df_main.loc[mainseq] = [claim, topic, label, str(dt.date()) , sources]
        print([claim, topic, label, str(dt.date()), sources])
        sources = ''
        previous = claim
        mainseq += 1
#x2 = vec.fit_transform(df_main['claim'])
#y2 = mlp.predict(x2)
#for i in range(0,x2.shape[0]):
#    print(df_main.loc[i][0],y2[i])

'''
politifact.csv
0.politifact_page, 1.claim, 2.claim_source, 3.claim_citation, 4.claim_label, 
5.date_published, 6.researched_by, 7.edited_by, 8.tags, 9.page_citation, 10.page_url, 11.page_is_first_citation

1.claim, 4.claim_label (barely_true, half-true, mostly-true, pants-fire, FALSE, TRUE), 5.date_published, 10.page_url
'''

print('Getting politifact.csv dataset...')
df_politifact = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/rumor-citation/politifact.csv')
df_politifact_pred = pd.DataFrame(columns=['claim'])
seq = 0
for i in range(0, len(df_politifact)):
    claim = str(df_politifact.loc[i].values[1])
    claim = normalize_text(claim)
    df_politifact_pred.loc[seq] = [claim]
    seq += 1

print('Fitting countvectorizer...')
x2 = vec.fit_transform(df_politifact_pred['claim'])
print('Prediction...')
y2 = mlp.predict(x2)
print('Predictions:')
seq = 0
previous = ''
sources = ''
for i in range(0, len(df_politifact)):
    claim = str(df_politifact.loc[i].values[1])
    topic = y2[i]
    if topic != 0:
        continue
    date = df_politifact.loc[i].values[5]
    dt = parse(date)
    #print('DATE: ',date, dt.date())
    label = df_politifact.loc[i].values[4]
    if label == 'pants-fire' or label == 'half-true' or label == 'FALSE':
        label = 'false'
    else:
        label = 'true'
    parsed_uri = urlparse(str(df_politifact.loc[i].values[10]))
    result = '{uri.netloc}'.format(uri=parsed_uri)
    result = str(result)
    result.replace('www.','')
    sources += str(result) + ';'
    if claim != previous:
        df_main.loc[mainseq] = [claim, topic, label, str(dt.date()) , sources]
        print([claim, topic, label, str(dt.date()), sources])
        sources = ''
        previous = claim
        mainseq += 1
        #print(seq)

df_main.to_csv('fakenews.csv')