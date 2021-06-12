# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:11:35 2021

@author: Salehin
"""
#import library that contains data
import cms_procedure as cp
#for data analysis and manipulation
import pandas as pd
import numpy as np
#for splitting to train test
from sklearn.model_selection import train_test_split
#for label encoding the categorical to numerical
from sklearn.preprocessing import LabelEncoder

#for visualization
import seaborn as sns
import matplotlib.pyplot as plt

#for machine learning model and gridsearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score

#for saving model
import pickle
import joblib
#to check latency of prediction
import time



#get attribute data in pandas dataframe
def get_attribute(procedure_id):
    return pd.DataFrame(cp.get_procedure_attributes(procedure_id))

#get outcome data in pandas dataframe
def get_proc_success(procedure_id):
    return pd.DataFrame(cp.get_procedure_success(procedure_id))

#get procedure outcome measures in pandas dataframe
def get_proc_outcomes(procedure_id):
    return pd.DataFrame(cp.get_procedure_outcomes(procedure_id))

#count total missing values
def count_missing(data):
    count=0
    for i in data.isnull().sum(axis=1):
        if i>0:
            count=count+1
    return count*100/len(data.index)   

#function for generating gridsearch results
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))     
#function for evaluating model and checking prediction time
def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- \tAccuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                     accuracy,
                                                                                     precision,
                                                                                     recall,
                                                                                     round((end - start)*1000, 1)))

if __name__ == "__main__":
    #I assume procedure id's are list of numbers that one can draw. For example a possible
    #procedure_id would look like list(range(1,2000)).
    ids=list(range(1,10000))
    attributes=get_attribute(ids)  # return a dataframe containing attribute values for specific procedure id 
    success=get_proc_success(ids) # return a dataframe of true/false values for specific procedure id
    outcomes=get_proc_outcomes(ids) # return a dataframe of outcome measure values for specific procedure id
    
    #merge attrbibutes and success to get the full dataset without granular features
    data=pd.concat([attributes,success],axis=1,join='inner')
    
    #My goal is to build a model that can predict success (0 or 1) of a procedure given it's attributes 
    #Observe and get basic information on dataset
    print(data.head(50))
    print(data.info())
    
    #count missing values and percentage of missing values. Gives a count on missing values across all columns
    print(data.isnull().sum())
    print(count_missing(data))
    #My Assumptions on missing values: They are less than 5 percents and they are randomly distributed. In other words
    # missing values do not have patterns in our dataset. This could be discovered by futher analysis too. For now, I will drop the missing values
    data.dropna(axis=0,inplace=True)
    #Show relationships between variables using histogram
    sns.pairplot(data=data)
    print(data.corr()) # show correlation score between pairs of variables. applicable to only numerical features
    
    #Checking the correlation plots, My assumption is there is no high colinearity between features. So I don't plan on dropping any of them yem
    
    print(data.describe())  #check range of the numerical variables (Also gives us idea on possible outliers)
    
    #Again my assumption is that there is no significant outliers on the dataset so I don't want to remove any columns for now
    #Label encoding categorical variables (convert categrocial to numerical values)
    for feature in data.dtypes(include='String').index:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature].astype(str))
    #splitting to features and target
    y=data.loc[:,['success']]   
    data.drop(['success'],inplace=True)    
    X_train,X_test,y_train,y_test=train_test_split(data,y,test_size=0.25) #splitting to train_test
    #Building random forest model for classifying outcome. 
    rf = RandomForestClassifier()
    #specify parameters for grid search
    parameters = {
    'n_estimators': [2**i for i in range(3, 10)],
    'max_depth': [2, 4, 8, 16, 32, 64]
    }
    cv = GridSearchCV(rf, parameters, cv=5) # Using 5 folds cross validation
    cv.fit(X_train,y_train)
    print_results(cv) 
    feat_imp = cv.best_estimator_.feature_importances_  # extract feature importance
    indices = np.argsort(feat_imp)
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.barh(range(len(indices)), feat_imp[indices], color='r', align='center') # bar chart that shows feature importance for each variable on descending order
    plt.show()    
    joblib.dump(cv.best_estimator_, 'C://ml_model.pkl') #saving the model on local drive
    evaluate_model('rf_model', cv.best_estimator_, X_test, y_test)     #evaluate the model on test dataset. it returns accuracy, prediction, recall and time to make prediction

    
        
        
    
    
    
    
    
    
    

    
    