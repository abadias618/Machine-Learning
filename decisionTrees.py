#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abdias baldiviezo
"""
import pandas as pd
import math
from sklearn.model_selection import train_test_split
"""
Simple Node class to build a Tree,
only points to child nodes, not parent
"""
class Node():
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
"""
@param takes the root node of a tree
Prints the Tree using recursion
"""       
def printTree(root):
    if isinstance(root.data,str):
        print("data",root.data)
    else:
        print("data",type(root.data))
    if root.left is not None:
        print("left data")
        printTree(root.left)
    if root.right is not None:
        print("right data")
        printTree(root.right)
    return
"""
@return pre-processed data for the IRIS DATASET
data is modified to fit the decision tree, columns can only have 2 variable types
target column can have any number of target variable type
"""
def loadAndPrepareIrisDataSet():
    url = "https://byui-cs.github.io/cs450-course/week01/iris.data"
    data = pd.read_csv(url)
    #here we check some aspects of the data to determine how to fit it in
    #our tree
    print("describe the data\n",data.describe(include='all'))
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
"""
@return pre-processed Car Evaluations dataset
"""
def loadAndPrepareCarEvaluationDataSet():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    data = pd.read_csv(url)
    data.rename(columns = {'vhigh':'buying' , 'vhigh.1':'maint' , '2':'doors' , '2.1':'persons' , 'small':'lug_boot' , 'low':'safety' , 'unacc':'class' }, inplace=True)
    print('raw data:\n',data)
    data = pd.get_dummies(data, columns= ["buying", "maint", "doors", "persons", "lug_boot", "safety"],prefix=["buying", "maint", "doors", "persons", "lug_boot", "safety"])
    #make sure the targets are in the last column
    data = data[data.columns[::-1]]
    replacing = {"safety_med": {1 : "yes", 0 : "no"}, 
                 "safety_low": {1 : "yes", 0 : "no"},
                 "safety_high": {1 : "yes", 0 : "no"},
                 "lug_boot_small": {1 : "yes", 0 : "no"},
                 "lug_boot_med": {1 : "yes", 0 : "no"},
                 "lug_boot_big": {1 : "yes", 0 : "no"},
                 "persons_more": {1 : "yes", 0 : "no"},
                 "persons_4": {1 : "yes", 0 : "no"},
                 "persons_2": {1 : "yes", 0 : "no"},
                 "doors_5more": {1 : "yes", 0 : "no"},
                 "doors_4": {1 : "yes", 0 : "no"},
                 "doors_3": {1 : "yes", 0 : "no"},
                 "doors_2": {1 : "yes", 0 : "no"},
                 "maint_vhigh": {1 : "yes", 0 : "no"},
                 "maint_med": {1 : "yes", 0 : "no"},
                 "maint_low": {1 : "yes", 0 : "no"},
                 "maint_high": {1 : "yes", 0 : "no"},
                 "buying_vhigh": {1 : "yes", 0 : "no"},
                 "buying_med": {1 : "yes", 0 : "no"},
                 "buying_low": {1 : "yes", 0 : "no"},
                 "buying_high": {1 : "yes", 0 : "no"}}
    data.replace(replacing, inplace=True)
    print('pre-processed data:\n',data,"\ncolumns:\n",data.columns)
    
    
    return data
"""
@return pre-processed Voting dataset
"""
def loadAndPrepareVotingDataSet():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
    data = pd.read_csv(url)
    data.rename(columns = {'republican':'class' , 'n':'handicapped-infants' , 'y':'water-project-cost-sharing' , 'n.1':'adoption-of-the-budget-resolution' , 'y.1':'physician-fee-freeze' , 'y.2':'el-salvador-aid' , 'y.3':'religious-groups-in-schools', 'n.2':'anti-satellite-test-ban' , 'n.3':'aid-to-nicaraguan-contras', 'n.4':'mx-missile' , 'y.4':'immigration', '?':'synfuels-corporation-cutback' , 'y.5':'education-spending', 'y.6':'superfund-right-to-sue' , 'y.7':'crime', 'n.5':'duty-free-exports' , 'y.8':'export-administration-act-south-africa' }, inplace=True)
    #make sure the targets are in the last column
    data = data[data.columns[::-1]]
    print('raw data:\n',data)
    #replace all missing data "?" with the element that happens the most in
    #the column either "y" or "n"
    col = data.columns
    #len(col)-1 because we don't want to include the last column which is the target
    for i in range(len(col)-1):
        x = data[col[i]].describe()
        data.loc[(data[col[i]] == "?"), col[i]] = x.top
    print('raw data:\n',data)
    return data
"""
@param data in general, column to separate results, "string" name of the column to be processed
"""
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
    for key in listOfOptions:
        listOfTargets.update({key:countTargets(listOfOptions[key],targetColumn)})
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
"""
Calculates entropy scores for every option and for the attribute in general
"""
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
    return lowestOptionEntropy , weightedAverage, resultDictByOption, resultDictByTarget
"""
calculates the minimuim score of all the available attribute columns
"""
def calculate(df):
                    
        columns = df.columns
        scores = []
        #variables to help return the minimum
        scoreTemp = 2
        minIndex = None
        for i in range(len(columns)-1):
            option, score, newData, target = calculateEntropyAndSeparateDf(df, df[columns[i]],columns[len(columns)-1])
            columnName = columns[i]
            scores.append([newData, option, score, columnName, target])
            #determine best to set as the root, store the minimum
            if scores[i][2] < scoreTemp:
                scoreTemp = scores[i][2]
                minIndex = i
        return Node(scores[minIndex])

"""
determines which is the value of the leaf nodes(taken from the targets)
"""
def determineLeaf(leafDictionary):
    temp = 0
    result = None
    for key in leafDictionary:
        if leafDictionary[key] > temp:
            result = key
            temp = leafDictionary[key]
    return Node(result)
"""
ID3 algorithm
"""
"""data comes in in the node array [newData, minBranch, entropy score, name of column, leafDictionary]"""
def id3(node):
    #print("\nNode Debug:",node.data[0].keys()," minBranch:",node.data[1]," entroypy:",node.data[2]," column to target:",node.data[3]," result:",node.data[4].keys())
    for key in node.data[0]:
        #when the data has only 1 data type
        if len(node.data[0]) == 1:
            if key == node.data[1]:
                #insert result
                node.left = determineLeaf(node.data[4][key])
                if node.right is None:
                    node.right = determineLeaf(node.data[4][key])
                #printTree(root)
            elif key != node.data[1]:
                #insert result
                node.right = determineLeaf(node.data[4][key])
                if node.left is None:
                    node.left = determineLeaf(node.data[4][key])
            return
        #when data has no more columns but the target column
        elif len(node.data[0][key].columns) == 1:
            if key == node.data[1]:
                #insert result
                node.left = determineLeaf(node.data[4][key])
                if node.right is None:
                    node.right = determineLeaf(node.data[4][key])
            elif key != node.data[1]:
                #insert result
                node.right = determineLeaf(node.data[4][key])
                if node.left is None:
                    node.left = determineLeaf(node.data[4][key])
            return
        #normal case both datasets are fine    
        #this else is what would be executed in the first and second iteration
        else:
            if key == node.data[1]:
                #findleft
                nodeLeft = calculate(node.data[0][key])
                #set node
                node.left = nodeLeft
                #cut data from new data set
                for x in nodeLeft.data[0]:
                    nodeLeft.data[0][x] = nodeLeft.data[0][x].drop(columns = nodeLeft.data[3])
                #cut data from current data
                #THIS IS COMMENTED OUT BECAUSE IT DOES NOT ALLOW 
                #ATTRIBUTE REPEATS IN THE SAME NODE LEVEL, IF DESIRED
                #IT CAN BE ENABLED
#                for y in node.data[0]:
#                    if nodeLeft.data[3] != y:
#                        node.data[0][y] = node.data[0][y].drop(columns = nodeLeft.data[3])
                #enter recursion
                id3(nodeLeft)
            elif key != node.data[1]:
                #findright
                nodeRight = calculate(node.data[0][key])
                #set node
                node.right = nodeRight
                #cut data
                for x in nodeRight.data[0]:
                    
                    nodeRight.data[0][x] = nodeRight.data[0][x].drop(columns = nodeRight.data[3])
                #cut data from current data
                #THIS IS COMMENTED OUT BECAUSE IT DOES NOT ALLOW 
                #ATTRIBUTE REPEATS IN THE SAME NODE LEVEL, IF DESIRED
                #IT CAN BE ENABLED
#                for y in node.data[0]:
#                    if nodeRight.data[3] != y:
#                        node.data[0][y] = node.data[0][y].drop(columns = nodeRight.data[3])
                #enter recursion
                id3(nodeRight)
    return
"""
Creates the root node of a Tree and uses the ID3 algo to build the Tree
"""
def buildTree(data):
    root = calculate(data)
    #print("\nOld Data:",root.data[0].keys()," minBranch:",root.data[1]," entroypy:",root.data[2]," column to target:",root.data[3]," result:",root.data[4].keys())
    #strip data
    for x in root.data[0]:
        root.data[0][x] = root.data[0][x].drop(columns = root.data[3])
    
    id3(root)    
    return root
"""
Uses recursion to visit the Tree in order to get the result
"""
def recurseForResult(dataRow, node, result):
    if result != None:
        return result
    # determine whether you go left tor right
    condition = node.data[1]
    #print("node column to target",node.data[3])
    if dataRow[node.data[3]] == condition:
        if node.left is not None and isinstance(node.left.data,str):
            result = node.left.data
        result = recurseForResult(dataRow, node.left, result)
    else:
        if node.right is not None and isinstance(node.right.data,str):
            result = node.right.data
        result = recurseForResult(dataRow, node.right, result)
    return result
"""
iterates thourgh each row of the panda array and builds an array of results
"""
def predict(data, root):
    result = []
    #iterate through all dataset
    for i in range(len(data)):
        #isolate row from pandas df
        rowInData = data.loc[i, : ]
        rowResult = recurseForResult(rowInData, root, None)
        #build result
        result.append(rowResult)
    return result
"""
@param two arrays to compare 
@return accuracy measured by number of matches / length of the data
"""
def accuracy(array1, array2):
    counter = 0
    for i in range(len(array1)):
        if array1[i] == array2[i]:
            counter+=1
    print('Correct matches: ',counter," out of ",len(array1))
    return counter/len(array1)    
""""
Main
uncomment the dataset you want to test
"""
def main():
    data = loadAndPrepareIrisDataSet()
    train_data, test_data, train_targets, test_targets = train_test_split(data, data["species"], test_size=0.3)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_targets = train_targets.reset_index(drop=True)
    test_targets = test_targets.reset_index(drop=True)
    root = buildTree(train_data)
    #printTree(root)
    predictions = predict(test_data, root)
    print("accuracy",accuracy(predictions,test_targets))
    #MIGHT TAKE A LONG TIME
#    data2 = loadAndPrepareCarEvaluationDataSet()
#    train_data2, test_data2, train_targets2, test_targets2 = train_test_split(data2, data2["class"], test_size=0.9)
#    train_data2 = train_data2.reset_index(drop=True)
#    test_data2 = test_data2.reset_index(drop=True)
#    train_targets2 = train_targets2.reset_index(drop=True)
#    test_targets2 = test_targets2.reset_index(drop=True)
#    root2 = buildTree(train_data2)
#    predictions2 = predict(test_data2, root2)
#    print("accuracy",accuracy(predictions2,test_targets2))
    #MIGHT TAKE A LONG TIME
#    data3 = loadAndPrepareVotingDataSet()
#    train_data3, test_data3, train_targets3, test_targets3 = train_test_split(data3, data3["class"], test_size=0.3)
#    train_data3 = train_data3.reset_index(drop=True)
#    test_data3 = test_data3.reset_index(drop=True)
#    train_targets3 = train_targets3.reset_index(drop=True)
#    test_targets3 = test_targets3.reset_index(drop=True)
#    root3 = buildTree(train_data3)
#    predictions3 = predict(test_data3, root3)
#    print("accuracy",accuracy(predictions3,test_targets3))
main()