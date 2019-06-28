import pandas as pd
from nltk.tokenize import sent_tokenize
import string
import pickle
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/articles1.csv',header=0,encoding='ISO-8859-1')
df2 = pd.DataFrame(columns=['id','title','publication','author','date','year','month','url','content'])
seq = 0
mlp = MLPClassifier()
mlp = pickle.load(open('../News-classification.sav', 'rb'))
for i in range(0,len(df)):
    '''id,title,publication,author,date,year,month,url,content'''
    title= df.loc[i].values[2]
    publication = df.loc[i].values[3]
    author = df.loc[i].values[4]
    ArticleDate = df.loc[i].values[5]
    Year = df.loc[i].values[6]
    Month = df.loc[i].values[7]
    URL = df.loc[i].values[8]
    text = df.loc[i].values[9]
    sentences = []
    try:
        title = title.encode('ascii', errors='ignore').decode("utf-8")
        print(title)
        mlp.predict()
        text = text.encode('ascii', errors='ignore').decode("utf-8")
        print(str(text))
    except:
        continue
    print('--------------------------------------------------------------------------')
