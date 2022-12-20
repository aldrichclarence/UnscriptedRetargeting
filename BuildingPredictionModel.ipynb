# Import Libraries
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import numpy as np
import pandas as pd
import numpy
import math
import csv
import tensorflow as tf

# Load data - Training Set
# Example: Pre-processed Data for Training (Stratified 10-fold cross-validation) (Data for 9th Iteration)
dataset = read_csv('all_features_c9_train.csv', header=0, index_col=0)
dataset = dataset.iloc[: , 1:662]
values = dataset.values

# Check first few rows of dataset
print(dataset.head())

# Load data - Validation Set
val_dataset = read_csv('all_features_c9_val.csv', header=0, index_col=0)
val_dataset = val_dataset.iloc[: , 1:662]
val_values = val_dataset.values

# Check first few rows of dataset
print(val_dataset.head())

# Split train and validation set
train = dataset
val = val_dataset

# Get only values in the dataframe
print(val.head())
train = train.values
val = val.values
print(val)

# Split input and output for both train and validation set
train_X, train_y = train[:, :660], train[:, -1]
val_X, val_y = val[:, :660], val[:, -1]

# Check data
print(train_X.shape, len(train_X), train_y.shape)
print(val_X.shape, len(val_X), val_y.shape)
print(train_X)
print(val_X)
print(val_y)

# Print Validation Set's Real Outputs
for i in range(0, val_y.shape[0]):
    print(int(val_y[i]))

# Store the output's initial version (before applying one hot encoding)
train_y_label = train_y
val_y_label = val_y
print(val_y_label)

# Apply one hot encoding to the output data
train_y = to_categorical(train_y)
val_y = to_categorical(val_y)

# Check variable
print(val_y)

# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 220, 3))
val_X = val_X.reshape((val_X.shape[0], 220, 3))
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)

# Create callback for early stopping and model checkpoint
keras_callbacks   = [
      #EarlyStopping(monitor='val_loss', mode='min', verbose=1, baseline=0.02),
      ModelCheckpoint('unscripted-retargeting-lstm-model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose = 1)
]


# Design network
model = Sequential()
model.add(Masking(mask_value=-10, input_shape=(None, train_X.shape[2])))
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(64, return_sequences = True))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(24, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Build network
history = model.fit(train_X, train_y, epochs=30, batch_size = 32, validation_data=(val_X, val_y), verbose=2, shuffle=True, callbacks=keras_callbacks)

# Plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
