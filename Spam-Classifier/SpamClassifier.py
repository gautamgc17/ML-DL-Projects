#!/usr/bin/env python
# coding: utf-8

# #### Importing the DataSet

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


messages = pd.read_csv('SMSSpamCollection' , sep='\t' , names = ['Label' , 'Message'])
messages.head()


# In[3]:


messages.shape


# In[4]:


messages.isnull().sum()


# #### Data Cleaning and Data Preprocessing

# In[5]:


import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[6]:


ps = PorterStemmer()
wn = WordNetLemmatizer()


# In[7]:


corpus = []

for i in range(messages.shape[0]):
    
    review = re.sub('[^a-zA-Z]' , ' ' , messages['Message'][i])
    review = review.lower()
    
    review = word_tokenize(review)
    reviews = [wn.lemmatize(word) for word in review if word not in stopwords.words('english')]
    
    reviews = ' '.join(reviews)
    corpus.append(reviews)


# In[8]:


len(corpus)


# In[9]:


corpus[5:10]


# #### Building a Vocab and Vectorization

# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4500)


# In[11]:


X = cv.fit_transform(corpus).toarray()


# In[12]:


X.shape


# In[13]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(messages['Label'])


# In[14]:


y.shape


# In[15]:


y


# #### Splitting data into Train and Test data

# In[40]:


from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)


# In[41]:


print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)


# #### Training model using Naive Bayes Classifier

# In[42]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()


# In[43]:


mnb.fit(X_train , y_train)


# In[44]:


y_pred = mnb.predict(X_test)


# #### Confusion Matrix and Accuracy

# In[45]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score


# In[46]:


c_matrix = confusion_matrix(y_test , y_pred)
c_matrix


# In[47]:


accuracy = accuracy_score(y_test , y_pred)
print('Accuracy Score:' , accuracy*100)


# In[ ]:




