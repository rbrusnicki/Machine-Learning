# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:20:41 2023
@author: Roberto Brusnicki
"""

import numpy as np
import matplotlib.pyplot as plt
import HeadTracker as HT 

# LED positions in relation to Oculus Reference System in Meters
Lo = 0.001 * np.array([[-79.842, -48.518, 49.992],   # led1   (x,y,z)  
                       [-79.736,   1.483, -0.007],   # led2
                       [-79.842,  51.482, 49.992],   # led3
                       [ 79.842, -48.518, 49.992],   # led4
                       [ 79.736,   1.483, -0.007],   # led5
                       [ 79.842,  51.482, 49.992]])  # led6

# Some positions
x_angles = 0.1 * np.array([range(  0, 250, 5)])  # In degrees   entre   0 e 25
y_angles =       np.array([range(-32,  32, 1)])  # In degrees   entre -30 e 30 
z_angles = 0.1 * np.array([range(-50,  50, 5)])  # In degrees   entre  -5 e  5

QUAT = HT.Calc.createQUAT(x_angles, y_angles, z_angles)
plt.figure()
plt.plot(QUAT)
        
# Some euler angles
x_pos = 1e-3 * np.array([range( -100,  100, 5)]) # In meters
y_pos = 1e-3 * np.array([range( -150,   50, 5)]) # In meters
z_pos = 1e-3 * np.array([range( -700, -500, 5)]) # In meters

POS = HT.Calc.createPOS(x_pos, y_pos, z_pos)
plt.figure()
plt.plot(POS)

# lazer spot position on monitor
laserXY = HT.Calc.create_laserXY(QUAT, POS)
plt.figure()
plt.scatter(laserXY.T[0], laserXY.T[1])

# Leds positions after rotation
ledsXY = HT.Calc.create_ledsXY(QUAT, POS, Lo)
plt.figure()
plt.scatter(ledsXY.T[0], ledsXY.T[1])
plt.scatter(ledsXY.T[10], ledsXY.T[11])

# save to csv file
np.savetxt('ledsXY.csv', ledsXY, delimiter=';')
np.savetxt('laserXY.csv', laserXY, delimiter=';')

print("laserXY.shape: ", laserXY.shape)
print("ledsXY.shape: ", ledsXY.shape)

# load array
# data = np.loadtxt('data.csv', delimiter=';')