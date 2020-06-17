#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:55:49 2020

@author: abaldiviezo
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

"""Decode teh number array from the review into text"""
def decode_review(text, data):
    # this is a dictionary mapping words to an integer index
    word_index = data.get_word_index()
    
    #some indices are reserved for a certain purpose
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2 #Unknown
    word_index["<UNUSED>"] = 3
    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    return ' '.join([reverse_word_index.get(i, '?') for i in text]), word_index
def trim(data, word_index):
    data = keras.preprocessing.sequence.pad_sequences(data, 
                                                      value = word_index["<PAD>"],
                                                      padding = 'post',
                                                      maxlen = 256)
    return data
def main(): 
    #imdb si included with keras
    #data comes pre-processed in a list of numbers as seen by the line below
    imdb = keras.datasets.imdb
    #split the data
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    #ANALYZE THE DATA
    #number of records in training
    #print("training entries: {} , labels: {}".format(len(train_data), len(train_labels)))
    #we can see the the first and second record are of different lengths, something needs to be done
    #print(len(train_data[0]), len(train_data[1]))
    
    print("\nbefore padding train",train_data[0])
    decoded_train, word_index_train = decode_review(train_data[0], imdb)
    #print literal from words
    print('\n',decoded_train)
    #trim the data
    train_data = trim(train_data,word_index_train)
    print("\nafter padding train",train_data[0])
    
    print("\nbefore padding test",test_data[0])
    decoded_test, word_index_test = decode_review(test_data[0], imdb)
    #print literal from words
    print('\n',decoded_test)
    #trim the data
    test_data = trim(test_data,word_index_test)
    print("\nafter padding test",test_data[0])
    
    #EMBED
    #input shape is the vocabulary count used for the movie reviews (10000 words)
    vocab_size = 10000
    
    model = keras.Sequential()
    #figure out vectors in 16 dimensions
    model.add(keras.layers.Embedding(vocab_size, 16))
    #flaten into a 1 dimmensional vector
    model.add(keras.layers.GlobalAveragePooling1D())
    # feed to a dense layer of 16 nodes since there were 16 dimmensions
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    #output into a 1 node layer which is a sigmoid
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    #print results
    model.summary()
    
    #COMPILE THE MODEL
    #give it an optimizer and loss function
    #given that our output will only have 2 values the binary_crossentropy is a good loss function
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    #CREATE VALIDATION SET
    #10000 outta 25000
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    
    #TRAIN THE MODEL
    
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)
    
    #EVALUATE AGAINST THE TEST LABELS
    results = model.evaluate(test_data, test_labels)
    print(results)
main()