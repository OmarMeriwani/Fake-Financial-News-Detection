import re
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

news = pd.read_csv("C:/Users/Omar/Documents/MSc Project/Datasets/uci-news-aggregator.csv")
print(news.head())

