# Fake Financial News Detection
These documents contains the source code of the MSc dissertation of Omar Meriwani to achieve masters degree in big data and text analytics in the University of Essex. <br /> 
The work includes five models to detect fake financial news using sentiment analysis, news sources checking, objectivity check, checking against existing news and a fact-checking method. Datasets have been created especially for this project in addition to the online available data sources. Sentiment analysis model has been done using deep learning model in and it has achieved 87% accuracy, while the objectivity check has not achieved significant results. News sources analysis problem has been dealt with as a traditional term frequency problem. <br /> 
The solution achieved 94% accuracy value. Due to the lack of enough data sources, the fact-checking solution has ended up in creating a dataset that is ready for fact-checking against any relational dataset of periodic values such as the stock market. Finally, the similarity check against existing news has achieved 76% accuracy value.<br /> 
## Requirements
* Stanford Core NLP.
* Tensorflow.
* Sklearn.
* Google news word2vec embeddings (could be downloaded from [here](https://github.com/mmihaltz/word2vec-GoogleNews-vectors))
