import pandas as pd
import re
df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/BusinessTitlesFull.csv',header=0)
df2 = pd.DataFrame(columns=['ID','TITLE','URL','PUBLISHER','CATEGORY','HOSTNAME','TIMESTAMP'])

for i in range(0, len(df)):
    title= str(df.loc[i].values[1])
    numbers = re.findall(r'\d+ points|\d+ point|\d+\%', title)
    if numbers != []:
        print(title)

