import csv
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#only to run once to download stopword data and wordnet for lemmatizing
#nltk.download('stopwords')
#nltk.download('wordnet')


#read data from file
def read_data(file):
    data = []
    first = True
    with open(file, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if first != True:
                data.append(line)
            first = False
    return data

#preprocess data
def clean_data(text):
    ps = nltk.PorterStemmer()
    sw = stopwords.words('english')
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in sw]
    return text

#vectorize data
def vectorize_data(training_data, testing_data):
    train_tweets_tokenized = []
    for row in training_data:
        train_tweets_tokenized.append(row[3])
    test_tweets_tokenized = []
    for row in testing_data:
        test_tweets_tokenized.append(row[3])

    tfidf_vect = TfidfVectorizer(analyzer=clean_data)
    tfidf_vect_fit = tfidf_vect.fit(train_tweets_tokenized)
    X_train = tfidf_vect_fit.transform(train_tweets_tokenized)
    X_test = tfidf_vect_fit.transform(test_tweets_tokenized)
    return X_train, X_test