import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('C:/Users/Omar/Documents/MSc Project/Datasets/Kaggle Competition/train.csv',header=0)
data = []
labels = []
prev = ''
kaggle = True
for i in range(0,len(df)):
    #claim,label,date,sources
    #sources =  str(df.loc[i][1]).replace('www','').replace('.','').split(';')
    #Kaggle
    sources =  str(df.loc[i][2]).replace(' ','')
    #print(sources)
    sources2 = []
    for s in sources:
        if s == '':
            continue
        else:
            sources2.append(s)
    if sources2 == []:
        continue
    if kaggle == False:
        sources = ' '.join(sources2)
    #print(sources)
    label = df.loc[i][4]
    if kaggle == False:
        if str(label).lower()== 'true':
            label = 1
        else:
            label = 0
    data.append(sources)
    labels.append(label)
print(data)
x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.4)

tfidf = TfidfVectorizer()
_ = tfidf.fit(x_train)
train_tfidf = tfidf.transform(x_train)
test_tfidf = tfidf.transform(x_test)
print ('Transforming final test dataset')

model1 = MLPClassifier(hidden_layer_sizes = (20,20,20), solver ='lbfgs', random_state=50)
model1.fit(train_tfidf,y_train)
print(model1.score(test_tfidf,y_test))

'''
model = Sequential()
model.add(Dense(18, input_dim=(train_tfidf.shape[1] )))
model.add(Dense(20))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='hinge', metrics=['accuracy'])
model.fit(train_tfidf, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(test_tfidf, y_test))
score = model.evaluate(test_tfidf, y_test, batch_size=128, verbose=2)
print ('SCORE: ',score[1])
'''
'''
Adam: 63%
Adam with two layers: 68%
lbfgs: 75%

Kaggle: 94.3%
'''
