#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:25:50 2020

@author: abaldiviezo
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class HardCodedClassifier():
    
    def predict(self, test_data):
        self.test_data = test_data
        classifier = GaussianNB()
        classifier.fit(self.train_data, self.train_targets)
        targets_predicted = classifier.predict(test_data)
        #we would return this value but instead we are going to return
        #hardcoded values
        hardCode  = ['Iris-setosa']
        hardCodedArray = [hardCode] * 150
        return hardCodedArray
    def compareResults(self, array1, array2):
        counter = 0
        for i in range(len(array1)):
            if array1[i] == array2[i]:
                counter+=1
        print('Correct matches: ',counter)
        return counter/len(array1)
    def fit(self, data, targets):
        self.data = data
        self.targets = targets
        train_data, test_data, train_targets, test_targets = train_test_split(self.data, self.targets, test_size=0.7)
        self.train_data = train_data
        self.train_targets = train_targets
        test_results = self.predict(test_data)
        accuracy = self.compareResults(test_targets, test_results)
        print ('Accuracy: ',accuracy)
        

url = "https://byui-cs.github.io/cs450-course/week01/iris.data"
data = pd.read_csv(url)
print(data)
features = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
targets = data[["species"]].to_numpy()
hardCodedClassifier = HardCodedClassifier()
hardCodedClassifier.fit(features, targets)

