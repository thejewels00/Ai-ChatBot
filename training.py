# import random
# import json
# import pickle
# import numpy as np
# import nltk #natural language toolkit library
# from nltk.stem import WordNetLemmatizer #i need this library to reduce the words to their root form
# from tensorflow.keras.models import Sequential #i need this library to create a sequential model
# from tensorflow.keras.layers import Dense, Activation, Dropout #i need this library to create the layers of the model
# from tensorflow.keras.optimizers import SGD #i need this library to create the optimizer of the model


# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open('intents.json').read())

# words = []
# classes = []
# documents = []
# ignore_letters = ['!', '?', ',', '.']


# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# #print(documents)

# words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# words = sorted(set(words))

# classes = sorted(set(classes))

# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))


# training = []
# output_empty = [0] * len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])



# random.shuffle(training)
# training = np.array(training)

# training_x = list(training[:, 0])
# training_y = list(training[:, 1])

# # neural network model
# model = Sequential()
# model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu')) #input layer
# model.add(Dropout(0.5)) #to prevent overfitting
# model.add(Dense(64, activation='relu')) #hidden layer
# model.add(Dropout(0.5)) #to prevent overfitting
# model.add(Dense(len(training_y[0]), activation='softmax')) #output layer

# Sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) #optimizer
# model.compile(loss='categorical_crossentropy', optimizer=Sgd, metrics=['accuracy'])
# model.fit(np.array(training_x), np.array(training_y), epochs=200, batch_size=5)
# model.save('chatbot.model')


import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

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

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training_x = []
training_y = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training_x.append(bag)
    training_y.append(output_row)

training_x = np.array(training_x)
training_y = np.array(training_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu')) # input layer
model.add(Dropout(0.5))    #    dropout layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation='softmax'))

Sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=Sgd, metrics=['accuracy'])
model.fit(training_x, training_y, epochs=200, batch_size=5)
hist = model.fit(training_x, training_y, epochs=200, batch_size=5)
model.save('chatbot.h5', hist)
print('Done')
