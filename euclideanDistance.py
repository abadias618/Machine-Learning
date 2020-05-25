#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:31:34 2020

@author: abaldiviezo
"""

import numpy as np
import math
#calculate euclidean distance between 2 points
def calculate_euclidean(e1, e2):
    return math.sqrt((e1[1]-e1[0])**2+(e2[1]-e2[0])**2)
def find_smallest(dictionary):
    
    return smallest
    
x = np.array([3, 6])
y = np.array([5, 2])
data = np.array([[2,3],[3, 4], [5, 7], [2, 7], [3, 2], [1, 2], [9, 3], [4, 1]])
animals = ["dog", "cat", "bird", "fish", "fish", "dog", "cat", "dog"]

#Compute the distance between x and every row in the data array. 
#Save each of these distances into a new list or numpy array.
x_and_data = {}
for i in range(len(data)):
    x_and_data[i] = list((animals[i],calculate_euclidean(x, data[i])))
#x_and_data = np.array(x_and_data)
print("Your list between x and the data \n", x_and_data)

smallest = find_smallest(x_and_data)

print("smallest: ",smallest)

    