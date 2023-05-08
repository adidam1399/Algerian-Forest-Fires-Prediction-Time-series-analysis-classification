
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import seaborn as sns

from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# PCA

def PCA_transform(data, no_of_components,date_column):
    """ Function to perform PCA transformation on data based on no.of components"""
    pca=PCA(n_components=no_of_components)
    train_data_PCA=pca.fit_transform(data[data.columns.difference(['Date'])])
    train_data_PCA_df=pd.DataFrame(train_data_PCA)
    # Adding date column to transformed dataframe
    train_data_PCA_df.insert(0,'Date',list(date_column))
    return train_data_PCA_df

# Corelated Features


def transform_corr(data,corr_index):
    """ Getting the data which has features mostly corelated with labels"""
    data_copy=data.copy(deep=True)
    data_copy=data_copy[corr_index]
    data_copy['Date']=data['Date']
    data_copy=data_copy.drop(['labels'],1)
    return data_copy



