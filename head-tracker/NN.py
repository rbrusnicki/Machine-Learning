# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:49:43 2023
@author: Roberto Brusnicki
"""

import tensorflow as tf
import numpy as np

# Load the input features and output labels as NumPy arrays
input_features = np.loadtxt('ledsXY.csv', delimiter=';')
output_labels = np.loadtxt('laserXY.csv', delimiter=';')

# Expand the input features up to the 4th power
expanded_features = np.power(input_features, np.arange(1, 5))

# Split the data into training, cross-validation, and testing sets
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for cross-validation
test_ratio = 0.1   # 10% for testing

num_samples = len(expanded_features)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)

# Shuffle the indices
indices = np.random.permutation(num_samples)
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train + num_val]
test_indices = indices[num_train + num_val:]

# Split the data based on the indices
train_features, train_labels = expanded_features[train_indices], output_labels[train_indices]
val_features, val_labels = expanded_features[val_indices], output_labels[val_indices]
test_features, test_labels = expanded_features[test_indices], output_labels[test_indices]

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(114,)),
    #tf.keras.layers.Dense(128, activation='relu', input_shape=(input_features.shape[1] * 4,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mean_absolute_error'])

# Train the model using the training set
model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data=(val_features, val_labels))

# Make predictions on the test set
predictions = model.predict(test_features)

# Calculate the mean error on the cross-validation set
cv_predictions = model.predict(val_features)
mean_error = np.mean(np.abs(cv_predictions - val_labels))
print("Mean Error on Cross-Validation Set:", mean_error)

# Save the predictions to a CSV file
np.savetxt('output_predictions.csv', predictions, delimiter=',')