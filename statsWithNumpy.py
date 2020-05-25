#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:05:11 2020

@author: abaldiviezo
"""
import numpy as np

class SimpleCorrelator():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        def x():
            return self.x
        def y():
            return self.y
    def __repr__(self):
        return 'X - Summary\n' + '\tsize :' + str(len(self.x))+', mean: ' + str(np.mean(self.x)) + ', variance: ' + str(np.var(self.x)) + ', stdev: ' + str(np.std(self.x)) + '\nY - Summary \n' + '\tsize :' + str(len(self.y)) + ', mean: ' + str(np.mean(self.y)) + ', variance: ' + str(np.var(self.y)) + ', stdev: ' + str(np.std(self.y))
    
    def data_summary(self):
        return ''.join(['X - Summary\n', '\tsize :' , str(len(self.x)) , 
              ', mean: ' , str(np.mean(self.x)) , 
              ', variance: ' , str(np.var(self.x)) ,
              ', stdev: ' , str(np.std(self.x)) ,
              '\nY - Summary \n', '\tsize :' , str(len(self.y)) , 
              ', mean: ' , str(np.mean(self.y)) , 
              ', variance: ' , str(np.var(self.y)) ,
              ', stdev: ' , str(np.std(self.y))])
    def correlation(self):
        return str(np.corrcoef(self.x, self.y)[0,1])
          

# sales
sales = np.array([464.37,520.38,531.62,532.25,605.11,649.96,
                  432.75,438.40,410.72,598.27,437.15,441.57,
                  873.65,517.76,540.26,619.77,794.61,617.84,
                  351.75,727.77,504.56,564.19,697.12,461.50,
                  847.23,240.27,597.32,846.17,703.07,718.93,
                  622.53,544.62,510.13,608.06,597.89,487.62,
                  544.40,459.84,372.08,614.82,510.07,738.32,
                  756.48,360.73,271.03,533.69,586.78,743.36,
                  471.53,579.69])
# temperatures
temps = np.array([73.75,66.56,87.54,78.79,82.35,80.50,
                  76.09,74.13,74.83,76.13,76.41,72.08,
                  86.30,72.82,75.10,81.44,84.11,85.19,
                  70.34,81.89,78.31,78.35,89.29,77.31,
                  84.41,71.94,80.34,90.42,83.31,83.32,
                  76.81,74.50,73.16,77.07,81.04,77.14,
                  75.20,70.03,72.63,80.29,66.07,82.50,
                  83.00,69.97,61.72,77.11,76.07,85.83,
                  72.72,75.16])
simpleCorrelatorObject = SimpleCorrelator(sales, temps)
summary = simpleCorrelatorObject.data_summary()
print(summary)
print(simpleCorrelatorObject)
print('Correlation coeficient: ' + simpleCorrelatorObject.correlation())