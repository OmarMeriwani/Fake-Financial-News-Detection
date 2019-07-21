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
