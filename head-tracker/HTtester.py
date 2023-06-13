# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:53:19 2023
@author: Roberto Brusnicki
"""

import numpy as np
import math
#import matplotlib.pyplot as plt
import HeadTracker as HT

np.set_printoptions(precision=3, suppress = True)

# LED positions in relation to Oculus Reference System in Meters
Lo =  np.array([[0.1, 0.2, 0.0],         # led1   (x,y,z)  
                [0.2, 0.3, 0.0],      # led2
                [0.3, 0.1, 0.0],   # led3
                [0.1, 0.1, 0.0]])  # led4


pitch = (math.pi/180) * 0
yaw   = (math.pi/180) * 0
roll  = (math.pi/180) * 0

Qtest = np.zeros((7,4), dtype=float)
for i in range(7):
    Qtest[i] = HT.Calc.angle2quat(pitch, yaw, roll, 'XYZ')


POStest =  np.array([[  0,   1,   -1],         #   (x,y,z)  
                     [  0,   1,   -2], 
                     [  0,   1,   -3],
                     [  0,  10,  -10],         #   (x,y,z)  
                     [  0, -10, -100], 
                     [ 10,  10, -100], 
                     [-10, -10, -100]]) 

laserXYtest = HT.Calc.create_laserXY(Qtest, POStest)

ledsXYtest = HT.Calc.create_ledsXY(Qtest, POStest, Lo)

INPUTtest = np.concatenate((laserXYtest, ledsXYtest), axis=1)

print("laserXYtest.shape: ", laserXYtest.shape)
print("ledsXYtest.shape: ", ledsXYtest.shape)
print("INPUTtest.shape: ", INPUTtest.shape)

# load array
LASERdata = np.loadtxt('laserXY.csv', delimiter=';')
LEDdata = np.loadtxt('ledsXY.csv', delimiter=';')
print("LASERdata.shape: ", LASERdata.shape)
print("LEDdata.shape: ", LEDdata.shape)
