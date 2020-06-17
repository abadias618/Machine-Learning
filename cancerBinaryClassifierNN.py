#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:25:53 2020

@author: abaldiviezo
"""

from __future__ import absolute_import, print_function

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Sequential


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def loadCancerData():
    #wisconsin breast cancer data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    data = pd.read_csv(url, header=None)
    data.columns = ["id","target","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]
    # we don't care about the id
    data.drop(columns =["id"], inplace=True)
    # make the targets binary (B or M) to (1 or 0)
    data.loc[(data["target"] == "B"), "target"] = 1
    data.loc[(data["target"] != 1), "target"] = 0
    #min-max scaling 
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_1 = scaler.fit_transform(data["1"].values.reshape(-1, 1))
    scaled_2 = scaler.fit_transform(data["2"].values.reshape(-1, 1))
    scaled_3 = scaler.fit_transform(data["3"].values.reshape(-1, 1))
    scaled_4 = scaler.fit_transform(data["4"].values.reshape(-1, 1))
    scaled_5 = scaler.fit_transform(data["5"].values.reshape(-1, 1))
    scaled_6 = scaler.fit_transform(data["6"].values.reshape(-1, 1))
    scaled_7 = scaler.fit_transform(data["7"].values.reshape(-1, 1))
    scaled_8 = scaler.fit_transform(data["8"].values.reshape(-1, 1))
    scaled_9 = scaler.fit_transform(data["9"].values.reshape(-1, 1))
    scaled_10 = scaler.fit_transform(data["10"].values.reshape(-1, 1))
    scaled_11 = scaler.fit_transform(data["11"].values.reshape(-1, 1))
    scaled_12 = scaler.fit_transform(data["12"].values.reshape(-1, 1))
    scaled_13 = scaler.fit_transform(data["13"].values.reshape(-1, 1))
    scaled_14 = scaler.fit_transform(data["14"].values.reshape(-1, 1))
    scaled_15 = scaler.fit_transform(data["15"].values.reshape(-1, 1))
    scaled_16 = scaler.fit_transform(data["16"].values.reshape(-1, 1))
    scaled_17 = scaler.fit_transform(data["17"].values.reshape(-1, 1))
    scaled_18 = scaler.fit_transform(data["18"].values.reshape(-1, 1))
    scaled_19 = scaler.fit_transform(data["19"].values.reshape(-1, 1))
    scaled_20 = scaler.fit_transform(data["20"].values.reshape(-1, 1))
    scaled_21 = scaler.fit_transform(data["21"].values.reshape(-1, 1))
    scaled_22 = scaler.fit_transform(data["22"].values.reshape(-1, 1))
    scaled_23 = scaler.fit_transform(data["23"].values.reshape(-1, 1))
    scaled_24 = scaler.fit_transform(data["24"].values.reshape(-1, 1))
    scaled_25 = scaler.fit_transform(data["25"].values.reshape(-1, 1))
    scaled_26 = scaler.fit_transform(data["26"].values.reshape(-1, 1))
    scaled_27 = scaler.fit_transform(data["27"].values.reshape(-1, 1))
    scaled_28 = scaler.fit_transform(data["28"].values.reshape(-1, 1))
    scaled_29 = scaler.fit_transform(data["29"].values.reshape(-1, 1))
    scaled_30 = scaler.fit_transform(data["30"].values.reshape(-1, 1))
    
    
    data["1"] =scaled_1
    data["2"] =scaled_2
    data["3"] =scaled_3
    data["4"] =scaled_4
    data["5"] =scaled_5
    data["6"] =scaled_6
    data["7"] =scaled_7
    data["8"] =scaled_8
    data["9"] =scaled_9
    data["10"] =scaled_10
    data["11"] =scaled_11
    data["12"] =scaled_12
    data["13"] =scaled_13
    data["14"] =scaled_14
    data["15"] =scaled_15
    data["16"] =scaled_16
    data["17"] =scaled_17
    data["18"] =scaled_18
    data["19"] =scaled_19
    data["20"] =scaled_20
    data["21"] =scaled_21
    data["22"] =scaled_22
    data["23"] =scaled_23
    data["24"] =scaled_24
    data["25"] =scaled_25
    data["26"] =scaled_26
    data["27"] =scaled_27
    data["28"] =scaled_28
    data["29"] =scaled_29
    data["30"] =scaled_30
    print("data\n", data)
    
    features = data[["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]]
    targets = data["target"]
    train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.7)
    return train_data, test_data, train_targets, test_targets

def main():
    train_data, test_data, train_targets, test_targets = loadCancerData()
    classifier = Sequential()
    # input_dim = 30features
    #layers 16,8,6
    #last alyer is sigmoid to classify 2 features B or M in our case
    classifier.add(Dense(units=16, activation = 'relu', input_dim = 30))
    classifier.add(Dense(units=8, activation = 'relu'))
    classifier.add(Dense(units=6, activation = 'relu'))
    classifier.add(Dense(units=1, activation = 'sigmoid'))
    #loss function and optimizer
    classifier.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy')
    classifier.fit(train_data, train_targets, batch_size=1, epochs=100)
    # predict
    results = classifier.predict(test_data)
    #normalize results
    for i in range(len(results)):
        if results[i] >= 0.5:
            results[i] = 1
        else:
            results[i] = 0
    #accuracy
    counter = 0
    accuracy = None
    test_targets = test_targets.to_numpy()
    for j in range(len(test_targets)):
        if results[j] == test_targets[j]:
            counter+=1
    accuracy = counter/len(test_targets)
    
    print("accuracy is: ",accuracy, "you got: ", counter, "correct answers out of: ", len(test_targets))
main()