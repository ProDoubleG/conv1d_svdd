import numpy as np
import pandas as pd
import tensorflow as tf

from os import listdir
from os.path import isfile, join

import config as c
config = c.GlobalConfig()

batch_size = config.BATCH_SIZE
final_dimension = config.FINAL_SPACE_DIMENSION

# INPUT : csv_address of data shape: (length_of_data, number_of_columns)
# OUTPUT : reshaped numpy form of the data
def reshape_by_batch(csv_address,batch_size=batch_size):
    datafile = pd.read_csv(csv_address) # read file
    data = data[:len(data) - len(data)%batch_size] # trim dataset to divisable lentgh
    #
    ...
    #
    data_list = np.array(data) # turn dataset into numpy array(for reshape)
    data_reshaped = data_list.reshape((int)(len(data_list)/batch_size),batch_size,data_list.shape[-1]) # reshape the dataset : (number_of_data, batch_size, dimension)
     
    return data_reshaped

# INPUT : folder directory of data
# OUTPUT : list of file names inside the directory
def gather_file_names_from_directory(directory_name):
    return [f for f in listdir(directory_name) if isfile(join(directory_name, f))]

# INPUT : folder directory of data
# OUTPUT : concatenated numpy form of file inside the directory
def concat_data_from_files(file_address):
    file_list = gather_file_names_from_directory(file_address)
    
    data_list = []
    for file_name in file_list:
        tmp = reshape_by_batch(file_address+file_name,batch_size)
        data_list.append(tmp)
    data_list = np.concatenate(data_list)
    
    return data_list

# define loss as the vector length since we are trying to map the data to point of origin
def vector_size_loss(y_true, y_pred):
    power = tf.constant([2.0])
    ones = tf.ones([final_dimension,], dtype=tf.float32)
    power = tf.math.multiply(ones, power)
    tmp = tf.pow(y_pred, power)
    tmp = tf.math.reduce_sum(tmp)
    vector_size = tf.sqrt(tmp)
    return vector_size

# function used to defined to calculate the vector length of the predictions
def vector_length(pred):
    vector_array = []
    for i in range(len(pred)):
        tmp = np.square(pred[i])
        tmp = np.sum(tmp)
        vector_size = np.sqrt(tmp)
        vector_array.append(vector_size)
    return vector_array

# set threshold by the multiple of trained_result's std
def anomaly_counter(value,mean,std, multiplier=2.5):
    cnt = 0
    if value >= (mean + multiplier*std):
        cnt = 1
    else:
        pass
    return cnt

# INPUT : count anomality inside the batch
# Print the anomality in numbers out of whole data length and represent it by percentage
def score_anomals(batch_data):
    for batch in batch_data:
        batch = np.expand_dims(batch, axis=0)
        predictions = encoder_model.predict(batch)
        vector = vector_length(predictions)
        cnt += anomaly_counter(vector,standard_mean,standard_std)
        del batch, predictions, vector
    print(cnt,"/",len(reshaped_batch),"   ",(cnt+0.0)/len(reshaped_batch)*100,"%")

# count anomnals
def anomaly_counter(value,mean,std):
    cnt = 0
    if value >= (mean +2.5*std):
        cnt = 1
    else:
        pass
    return cnt
