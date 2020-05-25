#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:24:30 2020

@author: abaldiviezo
"""
import pandas as pd
import math
class Node():
    def __init__(self,data):
       self.left = None
       self.right = None
       self.data = data
def preorderTraversal(node):
    if node is not None:
        #operate node here
        print(node.data)
        #go to left node
        preorderTraversal(node.left)
        #go to right node
        preorderTraversal(node.right)
def loadAndPrepareIrisDataSet():
    url = "https://byui-cs.github.io/cs450-course/week01/iris.data"
    data = pd.read_csv(url)
    #here we check some aspects of the data to determine how to fit it in
    #our tree
    print(data.describe(include='all'))
    #since the data isn't to varied we can split the lengths and widths
    #into 2 groups more than the median (50%) and less than the median
    #for each column
    
    #sepal_length
    data.loc[(data["sepal_length"] >= 5.800000), "sepal_length"] = "more_than_median"
    data.loc[(data["sepal_length"] != "more_than_median"), "sepal_length"] = "less_than_median"
    
    #sepal_width
    data.loc[(data["sepal_width"] >= 3.000000), "sepal_width"] = "more_than_median"
    data.loc[(data["sepal_width"] != "more_than_median"), "sepal_width"] = "less_than_median"
    
    #petal_length
    data.loc[(data["petal_length"] >= 4.350000), "petal_length"] = "more_than_median"
    data.loc[(data["petal_length"] != "more_than_median"), "petal_length"] = "less_than_median"
    
    #petal_width
    data.loc[(data["petal_width"] >= 1.300000), "petal_width"] = "more_than_median"
    data.loc[(data["petal_width"] != "more_than_median"), "petal_width"] = "less_than_median"
    
    print(data)
    return data
def separate(data, column, targetColumn):
    listOfOptions = {}
    listOfTargets = {}
    for i in range(len(column)):
        element = column[i]
        if element not in listOfOptions.keys():
            df = pd.DataFrame(columns = data.columns)
            listOfOptions.update({element:df})
            listOfOptions[element] = listOfOptions[element].append(data.loc[i, : ], ignore_index=True)
        elif element in listOfOptions.keys():
            listOfOptions[element] = listOfOptions[element].append(data.loc[i, : ], ignore_index=True)
    #
    for key in listOfOptions:
        listOfTargets.update({key:countTargets(listOfOptions[key],targetColumn)})
    #print("listofoptions", listOfOptions.keys())
    #print("listofotargets", listOfTargets.keys())
    return listOfOptions, listOfTargets

def countTargets(optionsDf, targetColumnName):
    listOfTargets = {}
    for i in range(len(optionsDf)):
        element = optionsDf[targetColumnName][i]
        if element not in listOfTargets.keys():
            listOfTargets.update({element: 0})
            listOfTargets[element] += 1
        elif element in listOfTargets.keys():
            listOfTargets[element] += 1
    
    return listOfTargets
def calculateEntropyAndSeparateDf(data, column, targetColumn):
    resultDictByOption, resultDictByTarget = separate(data, column, targetColumn)
    totalItemsDf = len(data)
    weightedAverage = 0
    lowestOptionEntropy = None
    for key in resultDictByOption:
        score = 0
        scoreTemp = 2
        totalItemsOption = len(resultDictByOption[key])
        for key2 in resultDictByTarget:
            for key3 in resultDictByTarget[key2]:
                score =- (resultDictByTarget[key2][key3]/totalItemsOption) * math.log2(resultDictByTarget[key2][key3]/totalItemsOption) 
        weightedAverage += (totalItemsOption/totalItemsDf) * score
        #little sort to see which one is the lowest
        if score < scoreTemp:
            scoreTemp = score
            lowestOptionEntropy = key
    return lowestOptionEntropy , weightedAverage, resultDictByOption

"""data comes in in an array [minBranch, entropy score, name of column]"""
def recur(node, data):
    if node.data[2]  == 0 or len(node.data) <= 1:
        return
    #insertleft
    columns = data[node.data[0]].columns
    print("\ncolumns", columns)
    scores = []
    #variables to help return the minimum
    scoreTemp = 2
    minIndex = None
    for i in range(len(columns)-1):
        option, score, newData = calculateEntropyAndSeparateDf(data, data[columns[i]],columns[len(columns)-1])
        columnName = columns[i]
        scores.append([option, score, columnName])
        #determine best to set as the root, store the minimum
        if scores[i][1] < scoreTemp:
            scoreTemp = scores[i][1]
            minIndex = i
    print("\nscores", scores)
    print("\nmin score", scores[minIndex])
    #chop off column
    #insert right
    
    return
def buildTree(data):
    #ROOT NODE
    #list of labels (columns)
    columns = data.columns
    print("\ncolumns", columns)
    scores = []
    #variables to help return the minimum
    scoreTemp = 2
    minIndex = None
    for i in range(len(columns)-1):
        option, score, newData = calculateEntropyAndSeparateDf(data, data[columns[i]],columns[len(columns)-1])
        columnName = columns[i]
        scores.append([newData, option, score, columnName])
        #determine best to set as the root, store the minimum
        if scores[i][2] < scoreTemp:
            scoreTemp = scores[i][2]
            minIndex = i
    print("\nscores", scores)
    print("\nmin score", scores[minIndex])
    #create the root
    root = Node(scores[minIndex])
    recur(root, data)
    #step 3 go into recursion with set separated
    
    #
    return root
def main():
    #myNode = Node(0)
    #myNode.left = Node(1)
    #myNode.right = Node(2)
    #preorderTraversal(myNode)
    data = loadAndPrepareIrisDataSet()
    #options = separate(data, data["sepal_length"])
    #print("options", options.keys())
    buildTree(data)
main()