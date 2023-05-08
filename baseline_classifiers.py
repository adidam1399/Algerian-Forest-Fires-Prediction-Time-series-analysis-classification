import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import seaborn as sns
import datetime
from utils import *
from preprocessing import *
from feature_selection import *
from train_models import *

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

# Nearest Means Classifiers


def Nearest_means_train(train_data_values,input_data_labels):
    """ Function to calculate Class means for training data"""
    indices_class_1=[]
    indices_class_2=[]
    #Getting indices for each class
    for i in range(len(input_data_labels)):
        if(input_data_labels[i]==0):
            indices_class_1.append(i)
        elif(input_data_labels[i]==1):
            indices_class_2.append(i)
    #Getting values for that indices which belongs to that class
    class_1_values=train_data_values.iloc[indices_class_1]
    class_2_values=train_data_values.iloc[indices_class_2]
    #Finding mean for each class
    class_1_mean=(np.mean(class_1_values,axis=0))
    class_2_mean=(np.mean(class_2_values,axis=0))
    #Returning the mean co-ordinates for each class    
    return class_1_mean,class_2_mean

def prediction_nearest_means(values,mean_list):
    """ Predictions based on nearest means"""
    labels_classified=[]
    for i in range(len(values)):
        if(np.linalg.norm(values.loc[i]-mean_list[0])<np.linalg.norm(values.loc[i]-mean_list[1])):
            labels_classified.append(0)
        else:
            labels_classified.append(1)
    return labels_classified

# Trivial Classifier

def trivial_system_classifier(train_labels,test_labels):
    """ Function to perform the trivial system classification"""
    unique_values, no_of_unique=np.unique(test_labels,return_counts=True)
    pred_labels=random.choices([0,1],weights=[(no_of_unique[0]/len(train_labels)),(no_of_unique[1]/len(train_labels))],k=len(test_labels))
    return accuracy_score(test_labels,pred_labels)*100, f1_score(test_labels,pred_labels),np.mean(f1_score(test_labels,pred_labels)),pred_labels
train_data_before_dropping_rows,train_labels_before_dropping=read_data("algerian_fires_train.csv")