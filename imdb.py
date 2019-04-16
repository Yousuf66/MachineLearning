# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:34:10 2018

@author: yousu
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb

(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

##each review is an array with each integer corresponding to a word in the dictionary 
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
#the first review
print(train_data[0])
##input to a neural netwoek must be of same length


#how to convert the integers back into words

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])

#we are padding the arrays to make sure they are of the same length ...making use of pad_sequences

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

len(train_data[0]), len(train_data[1])
print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

'''The first layer is an Embedding layer.
This layer takes the integer-encoded vocabulary and looks up the embedding vector for each wordindex.
These vectors are learned as the model trains.
 The vectors add a dimension to the output array.
 The resulting dimensions are: (batch, sequence, embedding).'''
 
 