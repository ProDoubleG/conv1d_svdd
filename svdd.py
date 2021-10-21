import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from os import listdir
from os.path import isfile, join

import config as c
config = c.GlobalConfig()

from functions import *

batch_size = config.BATCH_SIZE
final_dimension = config.FINAL_SPACE_DIMENSION

normal_train_path = "./normal/normal/"
concated_normal_train = concat_data_from_files(normal_train_path)

normal_valid_path = "./normal/normal_valid/"
concated_normal_valid = concat_data_from_files(normal_valid_path)

x_train = concated_normal_train 
y_train = np.zeros((len(x_train),batch_size))

x_valid = concated_normal_valid
y_valid = np.zeros((len(x_valid),batch_size))

inputs = tf.keras.Input(shape=(batch_size,3), name="input")
layer1 = tf.keras.layers.Conv1D(6, 3,  padding='same', use_bias=False)(inputs)
pooling1 = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(layer1)
layer2 = tf.keras.layers.Conv1D(12, 3,  padding='same', use_bias=False)(pooling1)
pooling2 = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(layer2)
layer3 = tf.keras.layers.Conv1D(24, 3,  padding='same',use_bias=False)(pooling2)
pooling3 = tf.keras.layers.MaxPool1D(pool_size=2, padding='same', name="encoder_output")(layer3)
flat_layer =  tf.keras.layers.Flatten()(pooling3)
output_layer = tf.keras.layers.Dense(final_dimension, activation=None, use_bias=False, kernel_initializer='ones')(flat_layer)

encoder_model = tf.keras.Model(inputs=inputs, outputs=output_layer)
callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1)
encoder_model.compile(optimizer="Adam", loss=vector_size_loss)
encoder_model.fit(x_train, y_train, epochs=500, verbose=0,validation_data=(x_valid, y_valid), callbacks=[callback])

predictions = encoder_model.predict(x_train)
standard_mean = np.mean(vector_length(predictions),axis=None)
standard_std = np.std(vector_length(predictions))

score_anomals_from_file_directory(normal_valid_path)
