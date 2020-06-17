#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:56:29 2020

@author: abaldiviezo
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def loadWheatData():
    #Measurements of geometrical properties of kernels belonging to three different varieties of wheat. A soft X-ray technique and GRAINS package were used to construct all seven, real-valued attributes.
	#targets Kama, Rosa, Canadian (1, 2, 3)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    data = pd.read_csv(url, header=None, delim_whitespace=True)
    data.columns = ["area","perimeter","compactness","kernel_length","kernel_width","asymmetry","groove_length","target"]
    
    #min-max scaling 
    scaler = MinMaxScaler(feature_range = (0,1))
    data["area"] = scaler.fit_transform(data["area"].values.reshape(-1, 1))
    data["perimeter"] = scaler.fit_transform(data["perimeter"].values.reshape(-1, 1))
    data["compactness"] = scaler.fit_transform(data["compactness"].values.reshape(-1, 1))
    data["kernel_length"] = scaler.fit_transform(data["kernel_length"].values.reshape(-1, 1))
    data["kernel_width"] = scaler.fit_transform(data["kernel_width"].values.reshape(-1, 1))
    data["asymmetry"] = scaler.fit_transform(data["asymmetry"].values.reshape(-1, 1))
    data["groove_length"] = scaler.fit_transform(data["groove_length"].values.reshape(-1, 1))
    print("data\n", data)
    features = data[["area","perimeter","compactness","kernel_length","kernel_width","asymmetry","groove_length"]]
    targets = data[["target"]]
    train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.7)
    return train_data, test_data, train_targets, test_targets
    
def main():
    train_data, test_data, train_targets, test_targets = loadWheatData()
    
    #train_targets = keras.utils.to_categorical(train_targets)
    print("shape: ", train_targets.shape)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=7))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    
    #optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.2, nesterov=True)
    
    #compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    #fit
    model.fit(train_data, train_targets, epochs=50, batch_size=1)
    
    #evaluate
    score = model.evaluate(test_data, test_targets, batch_size=1)
    print(score)
main()