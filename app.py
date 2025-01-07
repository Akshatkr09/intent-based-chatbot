#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st 


# In[9]:


# Step 1: Load the intents dataset
with open(r'C:\Users\KIIT\AICTE\intents.json') as file:
    data = json.load(file)


# In[10]:


print(data)


# In[11]:


# Step 2: Preprocess the data
patterns = []
responses = {}
labels = []

for intent in data:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']


# In[12]:


# Vectorize the patterns using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)


# In[13]:


# Encode labels
unique_labels = list(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
y = [label_to_index[label] for label in labels]


# In[14]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# Step 3: Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[16]:


def chatbot_response(user_input):
    user_input = user_input.lower()
    user_input_vectorize = vectorize.transform([user_input])
    predicted_index = model.predict(user_input_vectorize)[0]
    predict_label = index_to_label[predicted_index]
    return random.choice(responses.get(predicted_label, ["Sorry, I didn't understand that."]))

st.title("Intent-Based Chatbot")
st.write("This chatbot can respond to predefined intents. Type your message below!")

user_input = st.text_input("You:", "")


if user_input:
    bot_response = chatbot_response(user_input)
    st.text_area("Chatbot:", bot_response, height=100)

    


# In[ ]:




