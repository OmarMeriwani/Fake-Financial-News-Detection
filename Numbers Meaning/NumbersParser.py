import pandas as pd
import re
from stanfordcorenlp import StanfordCoreNLP
import os
import sys
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/BusinessTitlesFull.csv',header=0)
for i in range(0, 100000):
    title= str(df.loc[i].values[1])
    points = re.findall(r'stock|stocks|share|shares', title)
    dollars = []
    #dollars = re.findall(r'\d+\%|\$\d+|\£\d+|\€\d+|\d+\%|\$\d+(\.(\d+))?|\£\d+|\€\d+', title)
    #dollars = re.findall(r'\$\d+\.?\d+?\ ?B?', title)
    '''B, Billion, billion, bln, million, M, mln, thousnd'''
    if points != []:
        print('POINTS:', title, points)
    if dollars != []:
        print('DOLLARS:', title, dollars)