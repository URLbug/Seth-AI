import random

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Normalization
from tensorflow.keras.callbacks import Callback
import tensorflow as tf2

import string
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import nltk
import numpy as np

import pickle
import pandas as pd


nltk.download("stopwords")

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

word_list = []
classess = []
document = []

dataset = pd.read_json('dataset.json')

for i in dataset['intents']:
  for j in i['patterns']:
    words = preprocess_text(j)
    word_list.append(words)
    document.append((words,i['tag']))
    if i['tag'] not in classess:
      classess.append(i['tag'])

word_lists = [preprocess_text(word)
         for word in word_list]
word_lists = sorted(set(word_lists))

pickle.dump(word_lists, open('words.pkl', 'wb'))
pickle.dump(classess, open('classes.pkl', 'wb'))

training = []
output_emple = [0] * len(classess)

for document in document:
  bag = []
  word_pattern = document[0]
  word_pattern = [preprocess_text(i.lower()) for i in word_pattern]
  for word in word_list:
    bag.append(1) if word in word_pattern else bag.append(0)
  
  output_row = list(output_emple)
  output_row[classess.index(document[1])] = 1
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = np.array(list(training[:,0]))

def tf_tfidf(send_word):
  tk = tf2.keras.preprocessing.text.Tokenizer(num_words=17)
  tk.fit_on_texts(send_word)

  return tk.sequences_to_matrix(tk.texts_to_sequences(send_word), mode='tfidf')

train_x = tf_tfidf(word_lists)

model = Sequential()
model.add(Input((17,)))
model.add(Dropout(.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1,activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(train_x,train_y,epochs=200,batch_size=8, callbacks=[Callback()])
model.save('Seth.model', hist)
print(train_x.shape,train_y.shape)
