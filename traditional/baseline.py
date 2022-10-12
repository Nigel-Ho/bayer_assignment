import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string
#color = sns.color_palette()

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion,make_pipeline
from sklearn.base import TransformerMixin,BaseEstimator

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


def get_sentiment(text: str):
    
    """
    Function that uses NLTK.Vader to extract sentiment.
    Sentiment is a score that expresses how positive or negative a text is.
    The value ranges from -1 to 1, where 1 is the most positive value.
    Args:
        text (str): text to parse
    Returns:
        sentiment (float): polarity of the text
    """
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)['compound'] + 1

def get_nchars(text: str):
    """
    Function that returns the number of characters in a text.
    Args:
        text (str): text to parse
    Returns:
        n_chars (int): number of characters
    """
    return len(text)

def get_nsentences(text: str):
    """
    Function that returns the number of sentences in a text.
    Args:
         text (str): text to parse
    Returns:
       n_chars (int): number of sentences
    """
    return len(text.split("."))


class DummyTransformer(BaseEstimator, TransformerMixin):
    
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redudancy.
    """
    def __init__(self):
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return self

class Preprocessor(DummyTransformer):
    """
    Class used to preprocess text
    """
    def __init__(self, remove_stopwords: bool):
        self.remove_stopwords = remove_stopwords
        return None

    def transform(self, X=None):
        preprocessed = X.apply(lambda x: preprocess_text(x, self.remove_stopwords)).values
        return preprocessed

class SentimentAnalysis(DummyTransformer):
    """
    Class used to generate sentiment
    """
    def transform(self, X=None):
        sentiment = X.apply(lambda x: get_sentiment(x)).values
        return sentiment.reshape(-1, 1) # <-- note the reshape to transform a row vector into a column vector

class NChars(DummyTransformer):
    
    """
    Class used to count the number of characters
    """
    def transform(self, X=None):
        n_chars = X.apply(lambda x: get_nchars(x)).values
        return n_chars.reshape(-1, 1)

class NSententences(DummyTransformer):
    """
    Class used to count the number of sentences
    """
    def transform(self, X=None):
        n_sentences = X.apply(lambda x: get_nsentences(x)).values
        return n_sentences.reshape(-1, 1)

class FromSparseToArray(DummyTransformer):
    """
    Class used to transform a sparse matrix in a numpy array
    """
    def transform(self, X=None):
        arr = X.toarray()
        return arr


X = data['Sentence']
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23, test_size=0.2,stratify=y)

classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "SupportVectorMachine":SGDClassifier(loss='hinge',penalty='l2', alpha=1e-3, random_state=42)
}

vectorization_pipeline = Pipeline(steps=[
    ('preprocess', Preprocessor(remove_stopwords=True)), # the first step is to preprocess the text
    ('tfidf_vectorization', TfidfVectorizer()), # the second step applies vectorization on the preprocessed text
    ('arr', FromSparseToArray()), # the third step converts a sparse matrix into a numpy array in order to show it in a dataframe
])

tfidf = Pipeline(steps=[('vect', CountVectorizer()), 
                      ('tfidf', TfidfTransformer())
                      ])

features = [
  ('vectorization', tfidf),
  ('sentiment', SentimentAnalysis()),
]
combined = FeatureUnion(features)

for key, values in classifiers.items():
    
    clf = Pipeline([('feature',tfidf),
                          ('clf', values) 
                          ])
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print('{}  accuracy: {:4.4f}'.format(key,np.mean(predicted == y_test)))
    # 输出f1分数，准确率，召回率等指标
    # print(metrics.classification_report(y_test, predicted, target_names=categories))

# BernoulliNB  accuracy: 0.6481
# ComplementNB  accuracy: 0.6852
# MultinomialNB  accuracy: 0.6481
# KNeighborsClassifier  accuracy: 0.6111
# DecisionTreeClassifier  accuracy: 0.6111
# RandomForestClassifier  accuracy: 0.7222
# LogisticRegression  accuracy: 0.7407
# MLPClassifier  accuracy: 0.6852
# AdaBoostClassifier  accuracy: 0.7037
# SupportVectorMachine  accuracy: 0.7222