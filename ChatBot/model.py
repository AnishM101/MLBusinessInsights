import pickle
import json
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_json('ChatBot/data.json')
df = pd.json_normalize(df['intents'])

df = df.explode('patterns').explode('response')[['tag', 'patterns', 'response']]

label_encoder = LabelEncoder()
df['encoded_intent'] = label_encoder.fit_transform(df['tag'])

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

X = df['patterns']
y = df['encoded_intent'].values
pipeline.fit(X, y)

def respond(user_input):
    input_vector = pipeline.transform([user_input])
    predicted_class_index = pipeline.predict(input_vector)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_class_index])[0]

    response = df[df['tag'] == predicted_intent]['response'].values[0]
    return response

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)