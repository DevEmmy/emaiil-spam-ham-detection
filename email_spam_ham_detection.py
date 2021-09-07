#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd



# In[2]:


df = pd.read_csv('spam_ham_dataset.csv')


# In[3]:


df


# In[4]:


df.drop(['Unnamed: 0', 'label'], axis=1)


# In[12]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


# In[18]:


model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])


# In[19]:


X = df.text
y= df.label_num


# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=12)


# In[21]:


model.fit(X_train, y_train)


# In[29]:


model.score(X_test, y_test)



# In[40]:


def Check_Email_Spam_Ham(email):
    pred = model.predict(email)
    for p in pred:
        if p == '1':
            print('This email is a Spam')
        else:
            print('This email is not a Spam')

email_text = input('Please Enter the Mail Message : ')
text = []
text.append(email_text)
Check_Email_Spam_Ham(text)

