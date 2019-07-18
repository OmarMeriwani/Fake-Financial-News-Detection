from nltk.corpus import wordnet
from operator import __and__
def getVerbs(verb):
    synonyms = []
    for syn in wordnet.synsets(verb):
        for l in syn.lemmas():
            synonyms.append(l.name())
    #print(set(synonyms))
    return set(synonyms)

