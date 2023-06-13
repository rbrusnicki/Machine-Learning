# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:03:29 2023

@author: rbrus
"""

import numpy as np
from math import pi
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set the print options to limit floats to 3 decimal places for np arrays
np.set_printoptions(precision=3, suppress=True)

def ang2quat(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cy * cp * cr - sy * sp * sr
    x = cy * cp * sr + sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.concatenate((w, x, y, z), axis=0)

def quat2ang(w, x, y, z):
    # Convert quaternion to Euler angles (XYZ order)
    pitch = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    yaw = np.arcsin(2 * (w * y - z * x))
    roll = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return np.concatenate((pitch, yaw, roll), axis=1)

def zscore_normalize_features(X,rtn_ms=True):
    """
    returns z-score normalized X by column
    Args:
      X : (numpy array (m,n)) 
    Returns
      X_norm: (numpy array (m,n)) input normalized by column
    """
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
        
# Create angles and quats

pitch = ( 40 * np.random.rand(1,20000) - 20 ) * pi/180
yaw   = ( 40 * np.random.rand(1,20000) - 20 ) * pi/180
roll  = ( 40 * np.random.rand(1,20000) - 20 ) * pi/180

# Convert Euler angles to quaternions
quat = ang2quat(pitch, yaw, roll)

###############################################################################

# Normalize input
X  = np.concatenate((pitch, yaw, roll), axis=0).T

# Expand the input features up to the 13th power
#X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = np.c_[X, X**2, X**3, X**4, X**5]

input_data, input_mean, input_sigma = zscore_normalize_features(X)
output_data = quat.T


# Split the data into training, cross-validation, and testing sets
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for cross-validation
test_ratio = 0.1   # 10% for testing

num_samples = len(input_data)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)

# Shuffle the indices
indices = np.random.permutation(num_samples)
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train + num_val]
test_indices = indices[num_train + num_val:]

# Split the data based on the indices
train_data, train_output = input_data[train_indices], output_data[train_indices]
val_data, val_output = input_data[val_indices], output_data[val_indices]
test_data, test_output = input_data[test_indices], output_data[test_indices]

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_data.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Create a History callback to manually record loss history
history = tf.keras.callbacks.History()

# Train the model
model.fit(train_data, train_output, epochs=100, batch_size=32, validation_data=(val_data, val_output), callbacks=[history])

# Evaluate the model on the validation set
val_loss = model.evaluate(val_data, val_output)
print("Validation loss:", val_loss)

# Test the model on the test set
test_loss = model.evaluate(test_data, test_output)
print("Testing loss:", test_loss)

# Compare predictions with actual values and calculate error statistics
predictions = model.predict(test_data)
mae = mean_absolute_error(test_output, predictions)
mse = mean_squared_error(test_output, predictions)
std = np.std(np.abs(test_output - predictions))
print("Mean Absolute Error (MAE):", "%.3f" % mae)
print("Mean Squared Error (MSE):", "%.3f" % mse)
print("Standard Deviation (STD) of Absolute Error:", "%.3f" % std)

# Compare predictions with actual values and calculate error statistics (in degrees)
a = quat2ang(test_output[:, :1],test_output[:, 1:2],test_output[:, 2:3],test_output[:, 3:]) * 180/pi
b = quat2ang(predictions[:, :1],predictions[:, 1:2],predictions[:, 2:3],predictions[:, 3:]) * 180/pi
print("")
mae = mean_absolute_error(a, b)
mse = mean_squared_error(a, b)
std = np.std(np.abs(a - b))
print("Mean Absolute Error (MAE) in dregrees:", "%.3f" % mae)
print("Mean Squared Error (MSE) in dregrees:", "%.3f" % mse)
print("Standard Deviation (STD) of Absolute Error  in dregrees:", "%.3f" %std)

# Access the loss values from the training history
loss_history = history.history['loss']

# Plot the losses
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()