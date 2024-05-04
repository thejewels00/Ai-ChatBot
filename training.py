import random
import json
import pickle
import numpy as np
import nltk #natural language toolkit library
from nltk.stem import WordNetLemmatizer #i need this library to reduce the words to their root form
from tensorflow.keras.models import Sequential #i need this library to create a sequential model
from tensorflow.keras.layers import Dense, Activation, Dropout #i need this library to create the layers of the model
from tensorflow.keras.optimizers import SGD #i need this library to create the optimizer of the model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

