
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import seaborn as sns


def add_features(data,feature_num,column_name_1,column_name_2,column_name_3,n):
    """ Function to create new features, which are avg, max and min (operation) of a feature for past n number of days """
    new_feat_avg=[]
    new_feat_max=[]
    new_feat_min=[]
    for i in range(len(data)):
        if (i>=0 and i<2*n):
            new_feat_avg.append(data.iloc[i,feature_num])
            new_feat_max.append(data.iloc[i,feature_num])
            new_feat_min.append(data.iloc[i,feature_num])
        elif (i%2==0):
            new_feat_avg.append(np.mean(data.iloc[i-n*2:i,feature_num]))
            new_feat_avg.append(np.mean(data.iloc[i-n*2:i,feature_num]))
            new_feat_max.append(np.amax(data.iloc[i-n*2:i,feature_num]))
            new_feat_max.append(np.amax(data.iloc[i-n*2:i,feature_num]))
            new_feat_min.append(np.amin(data.iloc[i-n*2:i,feature_num]))
            new_feat_min.append(np.amin(data.iloc[i-n*2:i,feature_num]))
    data[column_name_1]=new_feat_avg
    data[column_name_2]=new_feat_max
    data[column_name_3]=new_feat_min
    return data

def add_statistic_features_to_data(data):
    """ Function which adds statistics of previous days to the given data based on add_features function"""
    data_copy=data.copy(deep=True)
    data_copy=add_features(data_copy,2,'avg_temp_2','max_temp_2','min_temp_2',2)
    data_copy=add_features(data_copy,2,'avg_temp_3','max_temp_3','min_temp_3',3)
    data_copy=add_features(data_copy,5,'avg_rain_2','max_rain_2','min_rain_2',2)
    data_copy=add_features(data_copy,5,'avg_rain_3','max_rain_3','min_rain_3',3)
    data_copy=add_features(data_copy,3,'avg_RH_2','max_RH_2','min_RH_2',2)
    data_copy=add_features(data_copy,3,'avg_RH_3','max_RH_3','min_RH_3',3)
    data_copy=add_features(data_copy,4,'avg_Ws_2','max_Ws_2','min_Ws_2',2)
    data_copy=add_features(data_copy,4,'avg_Ws_3','max_Ws_3','min_Ws_3',3)
    return data_copy


# Data Normalization using Standard Scaler (Standardization)

def Scale_feature(data, data_columns,Scaletype):
    """ Function which performs Standardization scaling of all the columns (except the date column) """
    data_date=data['Date']
    data=data.drop(columns='Date')
    data=Scaletype.fit_transform(data)
    data=pd.DataFrame(data,columns=data_columns[1:])
    data.insert(0,'Date',data_date)
    return data