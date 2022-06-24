import sys
# sys.append('../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import string
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer




def preprocess(df, columns =['text']):
        """ Function to preprocess the data 
        """
        df = to_lower(df)
        # df = replace_emojis(df, columns)
        df = filter_url(df, columns= columns)
        df = filter_mentions_hashtags(df, columns= columns)
        df = remove_punctuation(df, columns= columns)
        df = filter_stopwords(df, columns= columns)
        print("Tokenization")
        df = make_tokens(df, columns = columns)
        print("stemming")
        df = stemming(df, columns= columns)
        print("lemmatize")
        df = lemmatization(df,columns= columns)
        return df

def make_tokens(df, columns = ['text']):
    tt = TweetTokenizer()
    for col in columns:
        df[col] = df[col].apply(tt.tokenize)
    return df

def export(df, filename, columns= ['text']):
    df[columns].to_csv('../../data/processed/', filename, '.csv')


def replace_emojis(df, columns= ['text']):
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
    for col in columns:
            for emoji in emojis.keys():
                    df['text'] = df['text'].replace(emoji, "EMOJI" + emojis[emoji])
    return df

def filter_url(df, columns = ['text']):
    for col in columns:
        df[col] = df[col].apply(lambda x: re.sub(r'https?://\S+|www\.\S+| http?://\S+','', str(x)))
    return df
    
def filter_mentions_hashtags(df, columns = ['text']):
    for col in columns:
        df[col] = df[col].apply(lambda x: re.sub(r'\@\S+|\#', '', str(x)))
    return df
def filter_stopwords(df, columns = ['text']):
    stop_words = set(stopwords.words('english'))
    for col in columns:
            df[col] = df[col].apply(lambda x: ' '.join(x for x in str(x).split() if x not in stop_words))
    return df
def remove_punctuation(df, columns = ['text']):
    for col in columns:
        df[col] = df[col].str.translate(str.maketrans('', '', string.punctuation))
    return df

def stemming(df, columns = ['text']):
    pstemmer = PorterStemmer()
    for col in columns:
        # df[col] = df[col].apply(lambda x: " ".join(nltk.PorterStemmer().stem(str(text)) for text in x.split()))
        df[col] = df[col].apply(lambda word_list: [pstemmer.stem(x) for x in word_list])
    return df
def lemmatization(df, columns = ['text']):
    lemmatizer = WordNetLemmatizer()
    for col in columns:
            df[col] = df[col].apply(lambda x: [lemmatizer.lemmatize(str(val)) for val in x])
    return df

def to_lower(df, columns = ['text']):
    for col in columns:
        df[col] = df[col].str.lower()
    return df