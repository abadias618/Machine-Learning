#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:42:12 2020

@author: abaldiviezo
"""
from __future__ import absolute_import, print_function

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


"""to see that the model is still going, since my computer is kindaslow and old :("""
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')  



"""Load the dataset"""        
def loadAutoMpgData():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    dataset = pd.read_fwf(url)
    dataset.columns = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
    dataset.drop(columns =["car name"], inplace=True)
    #pre-process
    #replace the missing data in the horsepower column with the average
    #print(np.mean(data["horsepower"]))
    #the mean is 104.40
    dataset.replace({"horsepower": {"?": 104.40}}, inplace=True)
    #split the data
    #train_dataset = dataset.sample(frac=0.8, random_state=0)
    #test_dataset = dataset.drop(train_dataset.index)
    #min-max scaling the multi-valued discrete
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_mpg = scaler.fit_transform(dataset["mpg"].values.reshape(-1, 1))
    scaled_cylinders = scaler.fit_transform(dataset["cylinders"].values.reshape(-1, 1))
    scaled_displacement = scaler.fit_transform(dataset["displacement"].values.reshape(-1, 1))
    scaled_horsepower = scaler.fit_transform(dataset["horsepower"].values.reshape(-1, 1))
    scaled_weight = scaler.fit_transform(dataset["weight"].values.reshape(-1, 1))
    scaled_acceleration = scaler.fit_transform(dataset["acceleration"].values.reshape(-1, 1))
    scaled_model = scaler.fit_transform(dataset["model year"].values.reshape(-1, 1))
    scaled_origin = scaler.fit_transform(dataset["origin"].values.reshape(-1, 1))
    dataset["mpg"] =scaled_mpg
    dataset["cylinders"] = scaled_cylinders
    dataset["displacement"] = scaled_displacement
    dataset["horsepower"] = scaled_horsepower
    dataset["weight"] = scaled_weight
    dataset["acceleration"] = scaled_acceleration
    dataset["model year"] = scaled_model
    dataset["origin"] =scaled_origin
    print('pre-processed data:\n',dataset)
    features = dataset[["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]]#.to_numpy()
    targets = dataset["mpg"]#.to_numpy()
    train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.7)
    return train_data, test_data, train_targets, test_targets



"""build """
def build_model(train_data, test_data, train_targets, test_targets):
    model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.columns)]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    #mean squared error for loss
    #mean absolute error and mean squared error for metrics
    model.compile(loss='mse',optimizer=optimizer, metrics=['mae','mse'])
    
    return model

def main():
    #load and prepare data
    train_data, test_data, train_targets, test_targets = loadAutoMpgData()
    #build the model
    model = build_model(train_data, test_data, train_targets, test_targets)
    #summary to see what's up
    model.summary()
    #validation set
    example_batch = train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)
    #early stop
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # create a history object to see how well we do
    history = model.fit(
                train_data, train_targets,
                epochs = 500, validation_split=0.2, verbose=0,
                callbacks=[early_stop, PrintDot()]
            )
    #check our results from the history object
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    #mean absolute error
    loss, mae, mse = model.evaluate(test_data, test_targets, verbose=0)
    print("Mean absolute error is: {:5.2f} MPG".format(mae))
    results = model.predict(test_data).flatten()
    r2 = r2_score(test_targets, results)
    print("Automobile mpg r^2 score", r2)
    #print(results)
main()

