from senticnet.senticnet import SenticNet

sn = SenticNet()
word = 'and'
concept_info = sn.concept(word)
polarity_value = sn.polarity_value(word)
polarity_intense = sn.polarity_intense(word)
moodtags = sn.moodtags(word)
semantics = sn.semantics(word)
sentics = sn.sentics(word)

#print(concept_info)
#print(polarity_value)
print(polarity_intense)
#print(moodtags)
#print(semantics)
#print(sentics)