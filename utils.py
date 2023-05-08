
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import seaborn as sns

# Function to read the data

def read_data(data):
    """ Function to read the dataframe and split the data and its labels"""
    Data_read=pd.read_csv(data,header='infer')
    data=Data_read.iloc[:,:-1]
    labels=Data_read.iloc[:,-1]
    return data, labels


# Performing the splitting of data into 5 folds for Cross validation

def cv_split(data,start_date_train_1,end_date_train_1,start_date_val, end_date_val,start_date_train_2,end_date_train_2):
    """ Function to split data into validation fold and training folds based on its start and end dates """
    data_copy=data.copy(deep=True)
    data_copy.set_index('Date',inplace=True)
    validation=data_copy[start_date_val:end_date_val]
    train_data_1=data_copy[start_date_train_1:end_date_train_1]
    train_data_2=data_copy[start_date_train_2:end_date_train_2]
    train_data=pd.concat([train_data_1,train_data_2])
    return train_data, validation