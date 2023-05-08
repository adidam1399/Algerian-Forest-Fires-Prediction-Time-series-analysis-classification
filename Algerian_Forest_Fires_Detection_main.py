
# Algerian Forest Fires Detection

# Importing the required libraries

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
from baseline_classifiers import *

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



# Reading the data

train_data,train_labels=read_data("algerian_fires_train.csv")
test_data,test_labels=read_data('algerian_fires_test.csv')
data_complete=pd.concat([train_data,test_data])
labels_complete=pd.concat([train_labels,test_labels])
data_complete.head()

# Visualizing the corelation between the features in the training data

plt.figure(figsize=(12,8))
data_corr=train_data.corr()
sns.heatmap(data_corr,annot=True)
plt.show()

# Plotting the histogram of all the features - (To check on the outliers)

train_data.hist(bins=50,figsize=(20,15))
plt.show()

# Preprocessing the data
# Adding new features (expanding the feature space) which are quadratic combinations of original features

quadratic_features=PolynomialFeatures()
data_complete_copy=data_complete.copy(deep=True)
data_complete_copy_date=data_complete_copy['Date']
train_feat=data_complete_copy.columns[1:]
data_complete_copy=quadratic_features.fit_transform(data_complete_copy.drop(['Date'],axis=1))
data_complete_copy_features=quadratic_features.get_feature_names(train_feat)
data_complete_copy=pd.DataFrame(data_complete_copy,columns=data_complete_copy_features)
data_complete_copy.insert(0,'Date',list(data_complete_copy_date))
data_complete_copy.head()

# Adding new features to the existing data - Temperature, Rain, RH and WS seem uncorelated with other features, So adding new features which reflect statistics of these features

data_copy=data_complete_copy.copy(deep=True)
data_copy=add_statistic_features_to_data(data_copy)
data_copy.head()

# standardization of train and test data seperately

data_train=data_copy[:184]
train_data_normalized=Scale_feature(data_train,data_train.columns,StandardScaler())
data_test=data_copy[184:]
test_data_normalized=Scale_feature(data_test,data_test.columns,StandardScaler())
test_data_normalized.drop(['1'],axis=1);

# Dropping last 3 dates in train set because they influence the test set

train_data_normalized.drop(train_data_normalized.tail(6).index,inplace=True)
train_data_normalized=train_data_normalized.drop(['1'],axis=1)
train_data_normalized.head()
train_labels.drop(train_labels.tail(6).index,inplace=True)

# ### Performing the splitting of data into 5 folds for Cross validation

# #### Splitting the data based on dates

train_start_dates_1=['19/06/2012','01/06/2012','01/06/2012','01/06/2012','01/06/2012']
train_end_dates_1=['31/07/2012','15/06/2012','03/07/2012','21/07/2012','31/07/2012']
val_start_date=['01/06/2012','19/06/2012','07/07/2012','25/07/2012','12/08/2012']
val_end_date=['15/06/2012','03/07/2012','21/07/2012','08/08/2012','28/08/2012']
train_start_dates_2=['01/08/2012','07/07/2012','25/07/2012','12/08/2012','01/08/2012']
train_end_dates_2=['28/08/2012','28/08/2012','28/08/2012','28/08/2012','08/08/2012']

# ### Feature selection and reduction - Performing PCA to get the best feature set

#  Principal Components Analysis-PCA

pca_full=PCA()
pca_full.fit(train_data_normalized[train_data_normalized.columns.difference(['Date'])])
plt.figure(figsize=(10,8))
plt.title("PCA Skree Plot")
plt.xlabel('No.of features')
plt.ylabel('variance contributed by each feature')
plt.plot(pca_full.explained_variance_ratio_)
plt.show()
plt.figure(figsize=(10,8))
plt.title("PCA cumulative sum plot")
plt.ylabel('cumulative variance')
plt.xlabel('No.of features')
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.show()
print('Cumulative sum of variance list: {0}'.format(np.cumsum(pca_full.explained_variance_ratio_)))


# Running the PCA on training data for no.of components selected ranging from 6 to 20 on all the models and choosing the best one among them based on cross validation accuracy and F1 score

def PCA_select_components(data,no_of_comp,labels):
    """ Function to split data into folds based on no.of components given for PCA"""
    K_fold_split_pca=[]
    train_data_PCA_df=PCA_transform(data,no_of_comp,data['Date'])
    train_data_PCA_df['Classes']=labels
    for i in range(len(train_start_dates_1)):
        # For features obtained from pca
        train_pca,val_pca=cv_split(train_data_PCA_df,train_start_dates_1[i],train_end_dates_1[i],val_start_date[i],val_end_date[i],train_start_dates_2[i],train_end_dates_2[i])
        K_fold_split_pca.append([train_pca,val_pca])
    return K_fold_split_pca

# #### Training the models upon changing the number of components chosen by PCA

# #### Training the Linear Perceptron Model on components [ranging from (6,21)] obtained from PCA



accuracy_variation_Perceptron=train_models_initial(train_data_normalized,Perceptron,'Perceptron')


range_val=[i for i in range(6,23)]
plt.plot(range_val,accuracy_variation_Perceptron)
plt.title("Variation of accuracy for perceptron with change in no.of featueres selected by PCA")
plt.xlabel("No.of components")
plt.ylabel("Accuracy")
plt.show()

# ### Training The Naive Bayes classifier with assumption of pdf of features as Gaussian distributed to check performance on validation set

accuracy_variation_Bayes=train_models_initial(train_data_normalized,GaussianNB,'Bayes Classifier')

range_val=[i for i in range(6,23)]
plt.plot(range_val,accuracy_variation_Bayes)
plt.title("Variation of accuracy for Bayer classifier with change in no.of featueres selected by PCA")
plt.xlabel("No.of components")
plt.ylabel("Accuracy")
plt.show()

# #### Training the Logistic Regression Model on components [ranging from (6,21)] obtained from PCA by varying the C (regularization parameter value)



train_models_logistic(train_data_normalized,LogisticRegression,'Logistic Regression')


# Training the K-Nearest Neighbours Model on components [ranging from (6,21)] obtained from PCA by varying the no.of neighbours

train_models_KNN(train_data_normalized,KNeighborsClassifier,'K-Nearest Neighbors')

# #### Training the Random Forests Model on components [ranging from (6,21)] obtained from PCA 

Accuracy_list_Random_forests=train_models_initial(train_data_normalized,RandomForestClassifier,'Random Forests')

range_val=[i for i in range(6,23)]
plt.plot(range_val,Accuracy_list_Random_forests)
plt.title("Variation of accuracy for Random Forests classifier with change in no.of featueres selected by PCA")
plt.xlabel("No.of components")
plt.ylabel("Accuracy")
plt.show()

# #### Training Neural Network Classifier to check performance on validation set (Model selection with different neural network architectures)

# ##### Architecture of 1 hidden layer with 40 units in the hidden layer


NN_acc_1=train_models_initial(train_data_normalized,MLPClassifier,'Neural Network',hidden_layer_size=(40))

range_val=[i for i in range(6,23)]
plt.plot(range_val,NN_acc_1)
plt.title("Variation of accuracy for ANN classifier (no.of hidden layers=1 and no.of hidden units=40) with change in no.of featueres selected by PCA")
plt.xlabel("No.of components")
plt.ylabel("Accuracy")
plt.show()


# ##### Architecture of 2 hidden layer with (50,10) units in the hidden layer

NN_acc_2=train_models_initial(train_data_normalized,MLPClassifier,'Neural Network',hidden_layer_size=(50,10))

range_val=[i for i in range(6,23)]
plt.plot(range_val,NN_acc_2)
plt.title("Variation of accuracy for ANN classifier (no.of hidden layers=2 and no.of hidden units=(50,10) with change in no.of featueres selected by PCA")
plt.xlabel("No.of components")
plt.ylabel("Accuracy")
plt.show()

# ##### Architecture of 2 hidden layer (35, 15) units
NN_acc_3=train_models_initial(train_data_normalized,MLPClassifier,'Neural Network',hidden_layer_size=(35,15))

range_val=[i for i in range(6,23)]
plt.plot(range_val,NN_acc_3)
plt.title("Variation of accuracy for ANN classifier (no.of hidden layers=2 and no.of hidden units=(35,15) with change in no.of featueres selected by PCA")
plt.xlabel("No.of components")
plt.ylabel("Accuracy")
plt.show()


# #### Training the Ridge classifier on pca components ranging from 6 to 21 

Ridge=train_models_initial(train_data_normalized,RidgeClassifier,'Ridge Classifier',hidden_layer_size=(35,15))

# #### Training the SVM with Linear, polynomial Kernel and RBF Kernel on pca components ranging from 6 to 21 for various values of C (regularization) and Gamma (Kernel coefficient)

train_models_SVM(train_data_normalized)

train_data_PCA_df=PCA_transform(train_data_normalized,15,train_data_normalized['Date'])
svc_linear_final=SVC(kernel='linear',gamma=0.0001,C=0.01)
train_final=train_data_PCA_df.copy(deep=True)
train_final=train_final.drop(['Date'],axis=1)
train_labels_final=train_labels.copy(deep=True)
svc_linear_final.fit(train_final,train_labels_final)

# ### Test

test_data_pca=PCA_transform(test_data_normalized,15,test_data_normalized['Date'])

pred_test=svc_linear_final.predict(test_data_pca.drop(['Date'],axis=1))

accuracy_score(pred_test,test_labels)*100


# #### As we can see, the best model from our model selection is given by using SVM with RBF Kernel with C as 100 and gamma as 0.001. The validation accuracy in this case is 93.45 % (For features obtained from PCA case)

# ### Feature selection and reduction - Using pearson corelation coefficent- checking which features are mostly corelated with labels and picking them

train_data_normalized['labels']=train_labels
train_transformed_corr=train_data_normalized.corr()
plt.figure(figsize=(22,15))
sns.heatmap(train_transformed_corr)
plt.show()
train_data_normalized[train_data_normalized.columns.difference(['Date'])];

# ### Picking the features which are mostly corelated with the labels


corr_labels=train_transformed_corr[(train_transformed_corr['labels']>0.5) |(train_transformed_corr['labels']<-0.5)]

corr_index=corr_labels.index

train_data_corr=transform_corr(train_data_normalized,corr_index)
train_data_corr
train_data_corr['Date']=train_data_normalized['Date']
train_data_corr['Classes']=train_labels



# Training the models on data obtained from choosing features selected from mostly corelated with labels

def K_fold_split_corr(data):
    K_corr=[]
    for i in range(len(train_start_dates_1)):
            # For features obtained from corr
            train_corr,val_corr=cv_split(data,train_start_dates_1[i],train_end_dates_1[i],val_start_date[i],val_end_date[i],train_start_dates_2[i],train_end_dates_2[i])
            K_corr.append([train_corr,val_corr])
    return K_corr


def train_models_corr(data,Model,Model_str,hidden_layer_size=(50)):
    np.warnings.filterwarnings('ignore')
    K_split_corr=K_fold_split_corr(data)
    accuracy,f1,mean_f1=Models_train(K_split_corr,Model,Hidden_layer_sizes= hidden_layer_size)
    print("The accuracy and mean F1 score for with {0} Model for data selection using corelation with labels is {1} and {2} ".format(Model_str,accuracy,mean_f1))

#  Training the Perceptron on data obtained from choosing features which has most corelation with labels

train_models_corr(train_data_corr,Perceptron,'Perceptron')

# Training the Logistic Regression (with varying regularization) on data obtained from choosing features which has most corelation with labels

train_models_corr_logistic_knn(train_data_corr,LogisticRegression,'Logistic Regression')

#  Training the Bayes Classifier on data obtained from choosing features which has most corelation with labels

train_models_corr(train_data_corr,GaussianNB,'Bayes Classifier')

# Training KNN by varying the number of neighbours on data obtained from choosing features which has most corelation with labels

train_models_corr_logistic_knn(train_data_corr,KNeighborsClassifier,'K Nearest Neighbors')

# Training Random Forests on data obtained from choosing features which has most corelation with labels

train_models_corr(train_data_corr, RandomForestClassifier,'Random Forests')

# Training Neural Network with various architectures on data obtained from choosing features which has most corelation with labels
# Architecture of 1 hidden layer with 40 units in the hidden layer
train_models_corr(train_data_corr,MLPClassifier,'Neural Network')

# Architecture of 2 hidden layers with (50,10)  units in the hidden layer
train_models_corr(train_data_corr,MLPClassifier,'Neural Network',(50,10))

# Architecture of 2 hidden layers with (35,15)  units in the hidden layer
train_models_corr(train_data_corr,MLPClassifier,'Neural Network',(35,15))

# Training the Ridge Classifier on data obtained from choosing mostly corelated features with labels 

train_models_corr(train_data_corr,RidgeClassifier,'Ridge Classifier')

# #### Training the SVM with Linear, polynomial Kernel and RBF Kernel on data obtained from choosing mostly corelated features with labels for various values of C (regularization) and Gamma (Kernel coefficient)

train_models_SVM_corr(train_data_corr)

# Feature selection and reduction - Picking the subset Using Sequential Feature Selection- Fitting on logistic regression

train_data_normalized_copy=train_data_normalized.copy(deep=True)
train_data_normalized_copy=train_data_normalized_copy.drop(['labels','Date'],axis=1)

sfs=SequentialFeatureSelector(LogisticRegression(),n_features_to_select= 7)
train_data_sfs=sfs.fit_transform(train_data_normalized_copy,train_labels)
indices=sfs.get_support(indices=True)

# Getting most important indices of features selected by SFS
train_data_sfs=pd.DataFrame(train_data_sfs)
train_data_sfs['Date']=train_data_normalized['Date']
train_data_sfs['Classes']=train_labels

# Training models on features obtained from SFS

def K_fold_split_sfs(data):
    K_sfs=[]
    for i in range(len(train_start_dates_1)):
            # For features obtained from corr
            train_corr,val_corr=cv_split(data,train_start_dates_1[i],train_end_dates_1[i],val_start_date[i],val_end_date[i],train_start_dates_2[i],train_end_dates_2[i])
            K_sfs.append([train_corr,val_corr])
    return K_sfs

def train_models_sfs(data,Model,Model_str,hidden_layer_size=(50)):
    np.warnings.filterwarnings('ignore')
    K_split_sfs=K_fold_split_sfs(data)
    accuracy,f1,mean_f1=Models_train(K_split_sfs,Model,Hidden_layer_sizes= hidden_layer_size)
    print("The accuracy and mean F1 score for with {0} Model for data selected using SFS is {1} and {2} ".format(Model_str,accuracy,mean_f1))

# #### Training Linear Perceptron on data obtained from Sequential Feature Selection

train_models_sfs(train_data_sfs,Perceptron,'Perceptron')

# ### Training the Logistic Regression (with varying regularization) on data obtained from SFS

train_models_sfs_logistic_knn(train_data_sfs,LogisticRegression,'Logistic Regression')

#  Training the Bayes Classifier on data obtained from choosing features with SFS


train_models_sfs(train_data_sfs,GaussianNB,'Bayes Classifier')

#  Training KNN by varying the number of neighbours on data obtained from SFS

train_models_sfs_logistic_knn(train_data_sfs,KNeighborsClassifier,'K Nearest Neighbors')

# Training Random Forests on data obtained from SFS

train_models_sfs(train_data_sfs,RandomForestClassifier,'Random Forests Classifier')

# Training Neural Network with various architectures on data obtained from SFS
# Architecture of 1 hidden layer with 40 units in the hidden layer

train_models_sfs(train_data_sfs,MLPClassifier,'Neural Network')

# Architecture of 2 hidden layers with (50,10) units in the hidden layer

train_models_sfs(train_data_sfs,MLPClassifier,'Neural Network',hidden_layer_size=(50,10))

# Architecture of 2 hidden layers with (30,10) units in the hidden layer

train_models_sfs(train_data_sfs,MLPClassifier,'Neural Network',hidden_layer_size=(30,10))

#  Training the Ridge Classifier on data obtained from SFS

train_models_sfs(train_data_sfs,RidgeClassifier,'Ridge Classifier')

# Training the SVM with Linear, polynomial Kernel and RBF Kernel on data obtained from SFS for various values of C (regularization) and Gamma (Kernel coefficient)

train_models_SVM_sfs(train_data_sfs)

# ### Finally, the best performing model after Model Selection is SVM (Linear Kernel) with C (regularization parameter) as 0.01, gamma as 0.0001 on the data obtained from choosing the mostly corelated features with labels. The accuracy on validation set is 92.823 % and the mean F1 score is 0.8854

# ## Testing the best performing model on the test set

test_data_corr=test_data_normalized.copy(deep=True)
test_data_corr['labels']=test_labels
test_data_corr=test_data_corr[corr_index]
test_data_corr=test_data_corr.drop(['labels'],1)

train_data_corr_copy=train_data_corr.copy(deep=True)
svc=SVC(kernel='linear',C=0.01,gamma=0.0001)
svc.fit(train_data_corr_copy.drop(['Classes','Date'],1),train_labels)

test_pred_final=svc.predict(test_data_corr)
print("The accuracy obtained on the test set with best performing model on the validation set is {0}".format(accuracy_score(test_pred_final,test_labels)*100))
print("The F1 scores of each label obtained on the test set with best performing model on the validation set is {0} ".format(f1_score(test_pred_final,test_labels,average=None)))
print("The Mean F1 score obtained on the test set with best performing model on the validation set is {0} ".format(np.mean(f1_score(test_pred_final,test_labels,average=None))))
conf_matirx_test=confusion_matrix(test_pred_final,test_labels)


print("The Confusion Matrix obtained on the test set with best performing model on the validation set")
ax=sns.heatmap(conf_matirx_test,annot=True)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()

# Testing the trivial model on the test set

accuracy_score_trivial,f1_score_trivial,mean_f1_trivial,pred_labels=trivial_system_classifier(train_labels_before_dropping,test_labels)

print("The accuracy obtained on the test set with trivial model is {0}".format(accuracy_score_trivial))
print("The F1 scores of each label obtained on the test set with trivial model is {0} ".format(f1_score_trivial))
print("The Mean F1 score obtained on the test set with trivial model is {0} ".format(np.mean(mean_f1_trivial)))
conf_matirx_test_trivial=confusion_matrix(pred_labels,test_labels)
print("The Confusion Matrix obtained on the test set with trivial model")
ax=sns.heatmap(conf_matirx_test_trivial,annot=True)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()

# ## Testing the baseline model on the test set
# ##### Coding up the Nearest Means Classifier

train_data_copy=train_data.copy(deep=True).drop(['Date'],axis=1)
test_data_copy=test_data.copy(deep=True).drop(['Date'],axis=1)
class_1_mean,class_2_mean=Nearest_means_train(train_data_copy,train_labels_before_dropping)
pred_baseline=prediction_nearest_means(test_data_copy,[class_1_mean,class_2_mean])

accuracy_score_baseline=accuracy_score(pred_baseline,pred_labels)*100
f1_score_baseline=f1_score(pred_baseline,test_labels,average=None)
mean_f1_baseline=np.mean(f1_score(pred_baseline,test_labels))

print("The accuracy obtained on the test set with baseline model is {0}".format(accuracy_score_baseline))
print("The F1 scores of each label obtained on the test set with baseline model is {0} ".format(f1_score_baseline))
print("The Mean F1 score obtained on the test set with baseline model is {0} ".format(np.mean(mean_f1_baseline)))
conf_matirx_test_baseline=confusion_matrix(pred_baseline,test_labels)
print("The Confusion Matrix obtained on the test set with baseline model")
sns.heatmap(conf_matirx_test_baseline,annot=True)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()