from stanfordcorenlp import StanfordCoreNLP
import os
import sys

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)
sent = '2nd-Half Turnaround seen for Urban Outfitters (URBN), Jefferies Says'
verb = 'Says'
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
        #Preceeded by amod and compound
        #print(i[0], i[1], i[2])
        if i[1] in verbindex and i[0] == 'nsubj':
            result.append([tags[i[2] - 1][0], tags[i[2] - 1][1], ners[i[2] - 1][1] ])
    #print(result)
    return result
WhoSaid(sent,verb)
#Track Worker Killed by Metro-North Train, MTA Says
#2nd-Half Turnaround Seen for Urban Outfitters (URBN), Jefferies Says
