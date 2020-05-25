#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:51:32 2020

@author: abaldiviezo
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

url = "https://byui-cs.github.io/cs450-course/week01/iris.data"
data = pd.read_csv(url)
print(data)
features = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
targets = data[["species"]].to_numpy()
train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.3)
print('test targets' , test_targets)
classifier = GaussianNB()
classifier.fit(train_data, train_targets)
targets_predicted = classifier.predict(test_data)
print(targets_predicted)