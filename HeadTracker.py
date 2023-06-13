# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:42:36 2023

@author: rbrus
"""

import numpy as np
import math

class Calc:
    
    def __init__(self, a=0):
        self.a = a
    
    def angle2quat(r1, r2, r3, order):
        #  Converts rotation angles to quaternion.
        #  Q = ANGLE2QUAT( R1, R2, R3 ) calculates the quaternion, Q, for given,
        #  R1, R2, R3 angles.  R1 is the angle of first rotation.  R2 is the 
        #  angle of second rotation.  R3 is the angle of third rotation.
        #  Q returns an 4 element array representing the quaternions. Q has its
        #  scalar number as the first element. Rotation angles are input in radians.    
        #
        #  Q = ANGLE2QUAT( R1, R2, R3, S ) calculates the quaternion, Q, for a
        #  given set of rotation angles, R1, R2, R3, and a specified rotation
        #  sequence, S.  
        #
        #  The default rotation sequence is 'XYZ' where the order of rotation
        #  angles for the default rotation are R1 = X Axis Rotation, R2 = Y Axis
        #  Rotation, and R3 = Z Axis Rotation. 
        #
        #  Examples:
        #
        #  Determine the quaternion from rotation angles, in XYZ order:
        #     yaw = 0.7854; 
        #     pitch = 0.1; 
        #     roll = 0;
        #     q = angle2quat( pitch, yaw, roll, 'XYZ' )
        
        quat = np.array([0.0, 0.0, 0.0, 0.0])
        cang = np.array([math.cos(r1/2), math.cos(r2/2), math.cos(r3/2)])
        sang = np.array([math.sin(r1/2), math.sin(r2/2), math.sin(r3/2)])
        
        quat[0] = cang[0]*cang[1]*cang[2] - sang[0]*sang[1]*sang[2]
        quat[1] = cang[0]*sang[1]*sang[2] + sang[0]*cang[1]*cang[2]
        quat[2] = cang[0]*sang[1]*cang[2] - sang[0]*cang[1]*sang[2]
        quat[3] = cang[0]*cang[1]*sang[2] + sang[0]*sang[1]*cang[2]
     
        return quat
    
    def quat2DCM(q):
        #Computes the DCM matrix as a function of the attitude quaternion.
        #
        # INPUT:
        #   q: quaternion               [-]
        #
        # OUTPUT:
        #   D_NB: DCM matrix that converts from Navigation reference system 
        #   to Body reference system    [-]
        #
        
        q0 = q[0]  # escalar component
        q1 = q[1]  # ex
        q2 = q[2]  # ey
        q3 = q[3]  # ez
        
        # 'Rotates' from Navigation Reference System to Body Reference System
        # If r1 is a vector writen in NRS, then D_NB*r1 is the same vector writen in BRS.
        # If r2 is a vector writen in BRS, then D_BN*r2 is the same vector writen in NRS.
        
        DCM_BN = np.array([[1-2*q2**2-2*q3**2,   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
                           [  2*(q1*q2+q0*q3), 1-2*q1**2-2*q3**2,   2*(q2*q3-q0*q1)],
                           [  2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1), 1-2*q1**2-2*q2**2]])
        
        #DCM_NB = np.transpose(DCM_BN)
        
        return DCM_BN
        q0 = q[0]  # escalar component
        q1 = q[1]  # ex
        q2 = q[2]  # ey
        q3 = q[3]  # ez
        
        # 'Rotates' from Navigation Reference System to Body Reference System
        # If r1 is a vector writen in NRS, then D_NB*r1 is the same vector writen in BRS.
        # If r2 is a vector writen in BRS, then D_BN*r2 is the same vector writen in NRS.
        
        DCM_BN = np.array([[1-2*q2**2-2*q3**2,   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
                       [  2*(q1*q2+q0*q3), 1-2*q1**2-2*q3**2,   2*(q2*q3-q0*q1)],
                       [  2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1), 1-2*q1**2-2*q2**2]])
        
        #DCM_NB = np.transpose(DCM_BN)
        
        return DCM_BN

    def createQUAT(x_ang, y_ang, z_ang):
        
        n = x_ang.size * y_ang.size * z_ang.size 
        QUAT  = np.zeros((n,4), dtype=float)
    
       #I = x_ang.size
        J = y_ang.size
        K = z_ang.size
    
        for i in range(x_ang.size):	
            for j in range(y_ang.size):
                for k in range(z_ang.size):
                    pitch = x_ang[0][i] * math.pi / 180
                    yaw   = y_ang[0][j] * math.pi / 180
                    roll  = z_ang[0][k] * math.pi / 180
                    
                    q = Calc.angle2quat(pitch, yaw, roll, 'XYZ')
                    
                    QUAT[i*J*K + j*K + k][0] = q[0]
                    QUAT[i*J*K + j*K + k][1] = q[1]
                    QUAT[i*J*K + j*K + k][2] = q[2]
                    QUAT[i*J*K + j*K + k][3] = q[3]
        
        return QUAT
    
    def createPOS(x_pos, y_pos, z_pos):
        n = x_pos.size * y_pos.size * z_pos.size 
        POS  = np.zeros((n,3), dtype=float)
        
       #I = x_pos.size
        J = y_pos.size
        K = z_pos.size
    
        for i in range(x_pos.size):	
            for j in range(y_pos.size):
                for k in range(z_pos.size):
                    POS[i*J*K + j*K + k][0] = x_pos[0][i]
                    POS[i*J*K + j*K + k][1] = y_pos[0][j]
                    POS[i*J*K + j*K + k][2] = z_pos[0][k]
        return POS
    
    def create_laserXY(QUAT, POS):
        n = QUAT.shape[0]
        XY  = np.zeros((n,2), dtype=float)
        
        for i in range(n):
            q0 = QUAT[i][0]
            q1 = QUAT[i][1] 
            q2 = QUAT[i][2] 
            q3 = QUAT[i][3] 
            
            posX = POS[i][0]
            posY = POS[i][1]
            posZ = POS[i][2]
            
            lambda_ = posZ / (2*q1**2 + 2*q2**2 - 1)
            
            XY[i][0] = posX + lambda_ * 2 * (q1*q3 + q0*q2)
            XY[i][1] = posY + lambda_ * 2 * (q2*q3 - q0*q1)
            
        return XY
    
    def create_ledsXY(QUAT, POS, Lo):
        n = QUAT.shape[0]
        m = Lo.shape[0]
        XY  = np.zeros((n,2*m), dtype=float)
       
        
        for i in range(n):
            D_OC = Calc.quat2DCM(QUAT[i])
            Lc_i = np.matmul(D_OC, Lo.T).T + POS[i]
                    
            for j in range(Lo.shape[0]):
                XY[i][2*j  ] = - Lc_i[j][0] / Lc_i[j][2]
                XY[i][2*j+1] = - Lc_i[j][1] / Lc_i[j][2]
                
                #TO DO: lidar quando com o caso em que o denominador fica proximo de zero
                    
        return XY