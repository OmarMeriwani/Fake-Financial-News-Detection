import pandas as pd
import re
from stanfordcorenlp import StanfordCoreNLP
import os
import sys

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/BusinessTitlesFull.csv',header=0)
df2 = pd.DataFrame(columns=['ID','TITLE','URL','PUBLISHER','CATEGORY','HOSTNAME','TIMESTAMP'])

for i in range(0, len(df)):
    title= str(df.loc[i].values[1])
    numbers = re.findall(r'\d+ points|\d+ point|\d+\%|\$\d+|\£\d+|\€\d+', title)
    if numbers != []:
        ners = scnlp.ner(title)
        for n in ners:
            if n[1] == 'ORGANIZATION':
                print(title, n[0])
                break

