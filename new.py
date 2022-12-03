import string
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import nltk

import pickle

from tensorflow.keras.models import load_model
import tensorflow as tf2

import numpy as np

import random

from dataset import dataset

model = load_model('Seth.model')

nltk.download("stopwords")
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

m = Mystem()

def removes(txt):
  trans = str.maketrans('', '',string.punctuation)
  return txt.translate(trans)

mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text

def tf_tfidf(send_word):
  tk = tf2.keras.preprocessing.text.Tokenizer(num_words=17)
  tk.fit_on_texts(send_word)

  return tk.sequences_to_matrix(tk.texts_to_sequences(send_word), mode='tfidf')

def plan_b(text):
  send_word = [i for i in preprocess_text(text)]

  bag = tf_tfidf(send_word)[0,:]
  
  res = model.predict(np.array([bag]))[0]
  error = .25
  result = [[i,r] for i,r in enumerate(res) if r > error]

  return_list = []
  for r in result:
    return_list.append({'intents': classes[r[0]], 'probabilty': str(r[1])})
  tag = return_list[0]['intents']
  loi = dataset['intents']

  for i in loi:
    if i['tag'] == tag:
      results = random.choice(i['response'])
      break
  return results

#prin = tf_tfidf([i for i in preprocess_text('спокойной ночи')])

print(plan_b('привет'))