#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:09:02 2020

@author: abaldiviezo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.neighbors import KNeighborsClassifier
class HardCodedClassifier():
    """in the array the data is arranged as [0]sepal_length  [1]sepal_width  [2]petal_length  [3]petal_width"""
    def calcDistance(self, x1, x2):
        return math.sqrt((x2[0]-x1[0])**2+(x2[1]-x1[1])**2+(x2[2]-x1[2])**2+(x2[3]-x1[3])**2)
    """Bubble sort"""
    def sortDistances(self, distances):
        n = len(distances)
        for i in range(n-1):
            for j in range(0, n-i-1):
                if distances[j][1] > distances[j+1][1] : 
                    distances[j], distances[j+1] = distances[j+1], distances[j]
    def mostCommon(self, sortedDistances,k):
        setosa=0
        versicolor=0
        virginica=0
        for i in range(k):
            if sortedDistances[i][0] == "Iris-setosa":
                setosa+=1
            elif sortedDistances[i][0] == "Iris-versicolor":
                versicolor+=1
            elif sortedDistances[i][0] == "Iris-virtrain_data, test_data, train_targets, test_targets = train_test_split(self.data, self.targets, test_size=0.7)ginica":
                virginica+=1
        if setosa > versicolor and setosa > virginica:
            return "Iris-setosa"
        elif versicolor > setosa and versicolor > virginica:
            return "Iris-versicolor"
        elif virginica > versicolor and virginica > setosa:
            return "Iris-virginica"
        elif virginica == versicolor and virginica > setosa:
            return "Iris-virginica"
        elif versicolor == setosa and versicolor > virginica :
            return "Iris-versicolor"
        elif virginica == versicolor and versicolor == setosa:
            return "Iris-setosa"
    def predict(self, test_data, k):
        self.test_data = test_data
        # 
        predictions = []
        for test_flower in self.test_data:
            distances= []
            for i in range(len(self.train_data)):
                distances.append([self.train_targets[i][0], self.calcDistance(test_flower, self.train_data[i])])
            self.sortDistances(distances)
            predictions.append([self.mostCommon(distances,k)])
    
        return predictions
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
        self.test_data = test_data
        self.test_targets = test_targets


        
url = "https://byui-cs.github.io/cs450-course/week01/iris.data"
data = pd.read_csv(url)
print(data)
features = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
targets = data[["species"]].to_numpy()
hardCodedClassifier = HardCodedClassifier()
hardCodedClassifier.fit(features, targets)
test_results = hardCodedClassifier.predict(hardCodedClassifier.test_data, k=15)
accuracy = hardCodedClassifier.compareResults(hardCodedClassifier.test_targets, test_results)
print ('Accuracy: ',accuracy)
#compare to an existing implementation
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(hardCodedClassifier.train_data, hardCodedClassifier.train_targets)
predictions = classifier.predict(hardCodedClassifier.test_data)
accuracyComparison = hardCodedClassifier.compareResults(hardCodedClassifier.test_targets, predictions)
print ('Accuracy sklearn: ',accuracyComparison)