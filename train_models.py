
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import seaborn as sns
from preprocessing import *
from feature_selection import *
from Algerian_Forest_Fires_Detection_main import *

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

# Defining a function which trains all the required Models

def Models_train(data,Model,Kernel='linear',no_of_neighbours=5,Hidden_layer_sizes=(50),C_val=1,Gamma=0.1):
    """ Function which trains the Models on data """
    if(Model==SVC):
        model=SVC(kernel=Kernel,gamma=Gamma,C=C_val)
    elif(Model==MLPClassifier):
        model=MLPClassifier(hidden_layer_sizes=Hidden_layer_sizes,max_iter=700,random_state=42)
    elif(Model==KNeighborsClassifier):
        model=Model(n_neighbors=no_of_neighbours)
    elif(Model==LogisticRegression):
        model=Model(C=C_val)
    else:
        model= Model()
    val_accuracies=[]
    val_f1_scores=[]
    for i in range(len(data)):
        train_labels=data[i][0]['Classes']
        data[i][0]=data[i][0].drop(['Classes'],axis=1)
        model.fit(data[i][0],train_labels)
        val_labels=data[i][1]['Classes']
        val_pred=model.predict(data[i][1].drop(['Classes'],axis=1))
        val_accuracies.append(accuracy_score(val_labels,val_pred))
        val_f1_scores.append(f1_score(val_labels,val_pred,average=None))
        data[i][0]['Classes']=train_labels
        data[i][1]['Classes']=val_labels
    return np.mean(val_accuracies)*100,np.mean(val_f1_scores,axis=0),np.mean(np.mean(val_f1_scores,axis=0))

 # Features from PCA

def train_models_initial(data,Model,Model_str,hidden_layer_size=(50)):
    """ Function for Training Perceptron, Random Forests, ANN, Bayes Classifier and Ridge Classifier """
    np.warnings.filterwarnings('ignore')
    accuracy_list=[]
    accuracy_max=0
    mean_f1_max=0
    comp_max=0
    for i in range(6,23):
        K_fold_split_pca=PCA_select_components(data,i,train_labels)
        accuracy,f1,mean_f1=Models_train(K_fold_split_pca,Model,Hidden_layer_sizes= hidden_layer_size)
        print("The accuracy and mean F1 score for {0} PCA components with {1} Model is {2} and {3} ".format(i,Model_str,accuracy,mean_f1))
        if(accuracy>accuracy_max):
            accuracy_max=accuracy
            mean_f1_max=mean_f1
            comp_max=i
        accuracy_list.append(accuracy)
    print("The maximum accuracy and corresponding mean F1 score using {0} is obtained for {1} number of components. The accuracy is {2} and corresponding mean F1 score is {3} ".format(Model_str,comp_max,accuracy_max,mean_f1_max))
    return accuracy_list

def train_models_logistic(data,Model,Model_str):
    """ Function to train logistic model for various C values"""
    np.warnings.filterwarnings('ignore')
    accuracy_max=0
    mean_f1_max=0
    comp_max=0
    c_max=0
    C_list=[0.05, 0.1, 0.7, 1,5,10]
    for i in range(6,22):
        for c in C_list:
            K_fold_split_pca=PCA_select_components(data,i,train_labels)
            accuracy,f1,mean_f1=Models_train(K_fold_split_pca,Model)
            print("The accuracy and mean F1 score for {0} PCA components with {1} Model with C as {2} is {3} and {4} ".format(i,Model_str, c,accuracy,mean_f1))
            if(accuracy>accuracy_max):
                accuracy_max=accuracy
                mean_f1_max=mean_f1
                comp_max=i
                c_max=c
    print("The maximum accuracy and corresponding mean F1 score using {0} with C as {1} is obtained for {2} number of components. The accuracy is {3} and corresponding mean F1 score is {4} ".format(Model_str,c_max,comp_max,accuracy_max,mean_f1_max))

def train_models_KNN(data,Model,Model_str):
    """ Function to train KNN for different number of neighbors"""
    np.warnings.filterwarnings('ignore')
    accuracy_max=0
    mean_f1_max=0
    comp_max=0
    n_max=0
    neighbour_list=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    for i in range(6,22):
        for neighbours in neighbour_list:
            K_fold_split_pca=PCA_select_components(data,i,train_labels)
            accuracy,f1,mean_f1=Models_train(K_fold_split_pca,Model)
            print("The accuracy and mean F1 score for {0} PCA components with {1} Model with No.of neighbours as {2} is {3} and {4} ".format(i,Model_str, neighbours,accuracy,mean_f1))
            if(accuracy>accuracy_max):
                accuracy_max=accuracy
                mean_f1_max=mean_f1
                comp_max=i
                n_max=neighbours
    print("The maximum accuracy and corresponding mean F1 score using {0} with No.of neighbours as {1} is obtained for {2} number of components. The accuracy is {3} and corresponding mean F1 score is {4} ".format(Model_str,n_max,comp_max,accuracy_max,mean_f1_max))

def train_models_SVM(data):
    """ Function to train SVM """
    np.warnings.filterwarnings('ignore')
    g_list=[0.0001,0.0005,0.001,0.01,0.1,1]
    C_list=[0.0001,0.001,0.01,0.1,1,10,100,500]
    Kernel_list=['linear','poly','rbf']
    accuracy_max_list=[]
    accuracy_max=0
    f1_max=0
    c_max=0
    gamma_max=0
    comp_max=0
    kernel_max=0
    for i in range(6,22):
        K_fold_split_pca=PCA_select_components(data,i,train_labels)
        for kernel in Kernel_list:
            for gamma in g_list:
                for c in C_list:
                    accuracy,f1,mean_f1=Models_train(K_fold_split_pca,SVC,Kernel=kernel,C_val=c,Gamma=gamma)
                    print("The accuracy for {0} PCA components with {1} Kernel for gamma as {2} and C as {3} is {4}".format(i,kernel,gamma,c,accuracy))
                    if(accuracy>accuracy_max):
                        accuracy_max=accuracy
                        c_max=c
                        gamma_max=gamma
                        comp_max=i
                        kernel_max=kernel
                        f1_max=mean_f1
    accuracy_max_list.append([c_max,gamma_max,comp_max,accuracy_max])
    print("The maximum accuracy using SVM is obtained for {0} number of components using {1} Kernel with C as {2} and gamma as {3}. The accuracy is {4} and corresponding f1 score is {5}".format(comp_max,kernel_max,c_max,gamma_max,accuracy_max,f1_max))


# For features selected which are mostly corelated with label


def train_models_corr_logistic_knn(data,Model,Model_str):
    np.warnings.filterwarnings('ignore')
    if Model!=KNeighborsClassifier:
        C_list=[0.05, 0.1, 0.7, 1,5,10]
    else:
        C_list=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    for c in C_list:
        K_split_corr=K_fold_split_corr(data)
        accuracy,f1,mean_f1=Models_train(K_split_corr,Model,no_of_neighbours=c)
        if Model!=KNeighborsClassifier:
            print("The accuracy and mean F1 score for with {0} Model with C as {1} for data selection using corelation with labels is {2} and {3} ".format(Model_str,c,accuracy,mean_f1))
        else:
            print("The accuracy and mean F1 score for with {0} Model with No.of neighbours as {1} for data selection using corelation with labels is {2} and {3} ".format(Model_str,c,accuracy,mean_f1))


def train_models_SVM_corr(data):
    np.warnings.filterwarnings('ignore')
    g_list=[0.0001,0.0005,0.001,0.01,0.1,1]
    C_list=[0.0001,0.001,0.01,0.1,1,10,100,500]
    Kernel_list=['linear','poly','rbf']
    accuracy_max=0
    f1_max=0
    c_max=0
    gamma_max=0
    kernel_max=0
    K_split_corr=K_fold_split_corr(data)
    for kernel in Kernel_list:
        for gamma in g_list:
            for c in C_list:
                accuracy,f1,mean_f1=Models_train(K_split_corr,SVC,Kernel=kernel,C_val=c,Gamma=gamma)
                print("The accuracy for with {0} Kernel for gamma as {1} and C as {2} is {3}".format(kernel,gamma,c,accuracy))
                if(accuracy>accuracy_max):
                    accuracy_max=accuracy
                    c_max=c
                    gamma_max=gamma
                    kernel_max=kernel
                    f1_max=mean_f1
    print("The maximum accuracy using SVM is obtained using {0} Kernel with C as {1} and gamma as {2}. The accuracy is {3} and corresponding f1 score is {4}".format(kernel_max,c_max,gamma_max,accuracy_max,f1_max))

# Features obtained from Sequential Feature Selection

def train_models_sfs_logistic_knn(data,Model,Model_str):
    np.warnings.filterwarnings('ignore')
    if Model!=KNeighborsClassifier:
        C_list=[0.05, 0.1, 0.7, 1,5,10]
    else:
        C_list=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    for c in C_list:
        K_split_sfs=K_fold_split_sfs(data)
        accuracy,f1,mean_f1=Models_train(K_split_sfs,Model,no_of_neighbours=c)
        if Model!=KNeighborsClassifier:
            print("The accuracy and mean F1 score for with {0} Model with C as {1} for data selection using SFS is {2} and {3} ".format(Model_str,c,accuracy,mean_f1))
        else:
            print("The accuracy and mean F1 score for with {0} Model with No.of neighbours as {1} for data selection using SFS is {2} and {3} ".format(Model_str,c,accuracy,mean_f1))

def train_models_SVM_sfs(data):
    np.warnings.filterwarnings('ignore')
    g_list=[0.0001,0.0005,0.001,0.01,0.1,1]
    C_list=[0.0001,0.001,0.01,0.1,1,10,100,500]
    Kernel_list=['linear','poly','rbf']
    accuracy_max=0
    f1_max=0
    c_max=0
    gamma_max=0
    kernel_max=0
    K_split_sfs=K_fold_split_sfs(data)
    for kernel in Kernel_list:
        for gamma in g_list:
            for c in C_list:
                accuracy,f1,mean_f1=Models_train(K_split_sfs,SVC,Kernel=kernel,C_val=c,Gamma=gamma)
                print("The accuracy for with {0} Kernel for gamma as {1} and C as {2} is {3}".format(kernel,gamma,c,accuracy))
                if(accuracy>accuracy_max):
                    accuracy_max=accuracy
                    c_max=c
                    gamma_max=gamma
                    kernel_max=kernel
                    f1_max=mean_f1
    print("The maximum accuracy using SVM is obtained using {0} Kernel with C as {1} and gamma as {2}. The accuracy is {3} and corresponding f1 score is {4}".format(kernel_max,c_max,gamma_max,accuracy_max,f1_max))