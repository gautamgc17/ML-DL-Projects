#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import joblib as jb
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



data = pd.read_csv('Language Detection.csv')

le = LabelEncoder()
y = data["Language"]
y = le.fit_transform(y)


model = jb.load('lang_det_model.pkl')
cv = jb.load('vectorizer.pkl')


def predict(text):
    text = re.sub(r'[!@#$(),\n"%^*?:;~`0-9]' , ' ' , text)
    text = text.replace('[ ]' , '')
    text = text.lower()
    
    text = ' '.join(word_tokenize(text))
    
    vect = cv.transform([text]).toarray()
    lang = le.inverse_transform(model.predict(vect))[0]
    return lang





