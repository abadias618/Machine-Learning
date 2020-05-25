#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:47:25 2020

@author: abaldiviezo
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
"""
DATA INFO
CAR car acceptability
. PRICE overall price
. . buying buying price
. . maint price of the maintenance
. TECH technical characteristics
. . COMFORT comfort
. . . doors number of doors
. . . persons capacity in terms of persons to carry
. . . lug_boot the size of luggage boot
. . safety estimated safety of the car
ATTRIBUTE INFO
| names file (C4.5 format) for car evaluation domain

| class values

unacc, acc, good, vgood

| attributes

buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.

"""
def loadCars():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    data = pd.read_csv(url)
    data.rename(columns = {'vhigh':'buying' , 'vhigh.1':'maint' , '2':'doors' , '2.1':'persons' , 'small':'lug_boot' , 'low':'safety' , 'unacc':'class' }, inplace=True)
    print('raw data:\n',data)
    #pre-process
    # buying, maint, and safety directly contribute to the result, so we encode them numerically
    replacing = {"buying": {"vhigh": 1, "high": 2, "med": 3, "low": 4}, 
                 "maint": {"vhigh": 1, "high": 2, "med": 3, "low": 4},
                 "doors": {"5more": 5},
                 "persons": {"more": 5},
                 "lug_boot": {"small": 1, "med": 2, "big": 3},
                 "safety": {"low": 1, "med": 2, "high": 3}}
    data.replace(replacing, inplace=True)
    print('pre-preocessed data:\n',data)
    return data
"""split the Cars data into training-testing"""
def prepareCars(data):
    features = data[["buying", "maint", "doors", "persons", "lug_boot", "safety"]].to_numpy()
    targets = data[["class"]].to_numpy()
    train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.7)
    return train_data, test_data, train_targets, test_targets
"""
Attribute Information:

    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
    
    Missing Attribute Values:  horsepower has 6 missing values
"""
def loadAutomobileMpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    data = pd.read_fwf(url)
    data.columns = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
    print('raw data:\n',data)
    #pre-process
    #replace the missing data in the horsepower column with the average
    #print(np.mean(data["horsepower"]))
    #the mean is 104.40
    data.replace({"horsepower": {"?": 104.40}}, inplace=True)
    #min-max scaling the multi-valued discrete
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_mpg = scaler.fit_transform(data["mpg"].values.reshape(-1, 1))
    scaled_cylinders = scaler.fit_transform(data["cylinders"].values.reshape(-1, 1))
    scaled_displacement = scaler.fit_transform(data["displacement"].values.reshape(-1, 1))
    scaled_horsepower = scaler.fit_transform(data["horsepower"].values.reshape(-1, 1))
    scaled_weight = scaler.fit_transform(data["weight"].values.reshape(-1, 1))
    scaled_acceleration = scaler.fit_transform(data["acceleration"].values.reshape(-1, 1))
    scaled_model = scaler.fit_transform(data["model year"].values.reshape(-1, 1))
    scaled_origin = scaler.fit_transform(data["origin"].values.reshape(-1, 1))
    data["mpg"] =scaled_mpg
    data["cylinders"] = scaled_cylinders
    data["displacement"] = scaled_displacement
    data["horsepower"] = scaled_horsepower
    data["weight"] = scaled_weight
    data["acceleration"] = scaled_acceleration
    data["model year"] = scaled_model
    data["origin"] =scaled_origin
    print('pre-preocessed data:\n',data)
    return data
"""split the Automobile MPG data into training-testing"""
def prepareAutomobileMpg(data):
    features = data[["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]].to_numpy()
    targets = data[["mpg"]].to_numpy()
    train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.7)
    return train_data, test_data, train_targets, test_targets  
"""
lengthy attribute info here:
https://archive.ics.uci.edu/ml/datasets/Student+Performance
"""  
def loadStudentPerformance():
    url = "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv"
    data = pd.read_csv(url, sep=';')
    print('raw data:\n',data)
    #pre-processing
    #most of the fileds are going to be Hot encoded or scaled(MinMax)
    #Min Max scaling
    scaler = MinMaxScaler(feature_range = (0,1))
    data["age"]  = scaler.fit_transform(data["age"].values.reshape(-1,1))
    data["Medu"]  = scaler.fit_transform(data["Medu"].values.reshape(-1,1))
    data["Fedu"]  = scaler.fit_transform(data["Fedu"].values.reshape(-1,1))
    data["traveltime"]  = scaler.fit_transform(data["traveltime"].values.reshape(-1,1))
    data["studytime"]  = scaler.fit_transform(data["studytime"].values.reshape(-1,1))
    data["failures"]  = scaler.fit_transform(data["failures"].values.reshape(-1,1))
    data["famrel"]  = scaler.fit_transform(data["famrel"].values.reshape(-1,1))
    data["freetime"]  = scaler.fit_transform(data["freetime"].values.reshape(-1,1))
    data["goout"]  = scaler.fit_transform(data["goout"].values.reshape(-1,1))
    data["Dalc"]  = scaler.fit_transform(data["Dalc"].values.reshape(-1,1))
    data["Walc"]  = scaler.fit_transform(data["Walc"].values.reshape(-1,1))
    data["health"]  = scaler.fit_transform(data["health"].values.reshape(-1,1))
    data["absences"]  = scaler.fit_transform(data["absences"].values.reshape(-1,1))
    data["G1"]  = scaler.fit_transform(data["G1"].values.reshape(-1,1))
    data["G2"]  = scaler.fit_transform(data["G2"].values.reshape(-1,1))
    #Hot Encoding
    data = pd.get_dummies(data, columns=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"], prefix=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"])
    print('pre-preocessed data:\n',data)
    return data
"""split the Automobile MPG data into training-testing"""
def prepareStudentPerformance(data):
    features = data[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'school_GP', 'school_MS', 'sex_F', 'sex_M', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
'guardian_mother', 'guardian_other', 'schoolsup_no', 'schoolsup_yes', 'famsup_no', 'famsup_yes', 'paid_no', 'paid_yes', 'activities_no', 'activities_yes', 'nursery_no', 'nursery_yes', 'higher_no', 'higher_yes', 'internet_no', 'internet_yes', 'romantic_no', 'romantic_yes']].to_numpy()
    targets = data[["G3"]].to_numpy()
    train_data, test_data, train_targets, test_targets = train_test_split(features, targets, test_size=0.7)
    return train_data, test_data, train_targets, test_targets
"""implment the KNN Classifier algoritm"""
def knnClassifier(train_data, train_targets, test_data, k):
    classifier = KNeighborsClassifier(k)
    classifier.fit(train_data, train_targets)
    predictions = classifier.predict(test_data)
    return predictions
"""implment the KNN Regressor algoritm"""
def knnRegressor(train_data, train_targets, test_data, k):
    classifier = KNeighborsRegressor(k)
    classifier.fit(train_data, train_targets)
    predictions = classifier.predict(test_data)
    return predictions
def main():
    #cars data set
    data = loadCars()
    train_data, test_data, train_targets, test_targets = prepareCars(data)
    predictions = knnClassifier(train_data, train_targets, test_data, 41)
    accu = accuracy_score(test_targets, predictions)
    print("cars accuracy", accu)
    #Automobile mpg data set
    data2 = loadAutomobileMpg()
    train_data2, test_data2, train_targets2, test_targets2 = prepareAutomobileMpg(data2)
    predictions2 = knnRegressor(train_data2, train_targets2, test_data2, 19)
    r2 = r2_score(test_targets2, predictions2)
    print("Automobile mpg r^2 score", r2)
    #Student Performance data set
    data3 = loadStudentPerformance()
    train_data3, test_data3, train_targets3, test_targets3 = prepareStudentPerformance(data3)
    predictions3 = knnRegressor(train_data3, train_targets3, test_data3, 19)
    r2_3 = r2_score(test_targets3, predictions3)
    print("Student Final Test r^2 score", r2_3)
    
    
main()