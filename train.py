import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
import pyfiglet
from typing_extensions import Annotated
import inquirer
from datetime import datetime, date 

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from joblib import dump,load
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from eli5 import show_weights
from eli5 import show_prediction



def randomforest():
    type_2_diabetes_data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

    #looking at the headers of the dataset
    type_2_diabetes_data.head(5)

    #looking at all the statistical data from the dataset
    type_2_diabetes_data.describe()

    type_2_diabetes_data.info()

    #heatmap visulisation to see corrilations
    sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

    split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
    for train_indices, test_indices in split.split(type_2_diabetes_data,type_2_diabetes_data[["Diabetes_012","Sex","Age"]]):
        strat_train_set = type_2_diabetes_data.loc[train_indices]
        strat_test_set = type_2_diabetes_data.loc[test_indices]

    #Stratified test set    
    strat_test_set
    #Stratified train set   
    strat_train_set

    strat_train_set.info()


    X = strat_train_set.drop(['Diabetes_012'], axis=1)
    y = strat_train_set[['Diabetes_012']]
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X)
    Y_data = y.to_numpy()
    #random forest set up
    clf = RandomForestClassifier()

    param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

    grid_search = GridSearchCV(clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
    grid_search.fit(X_data,Y_data)

    final_clf = grid_search.best_estimator_


    X_test = strat_test_set.drop(['Diabetes_012'], axis=1)
    Y_test = strat_test_set[['Diabetes_012']]
    scaler = StandardScaler()
    X_data_test = scaler.fit_transform(X_test)
    Y_data_test = Y_test.to_numpy()

    final_clf.score(X_data_test,Y_data_test)

    #exporting file
    dump(final_clf,filename="clf_random_forest_model_First.joblib")

    #importing file
    loaded_model = load(filename="clf_random_forest_model_First.joblib")\
        
    joblib_y_preds = loaded_model.predict(X_test)
    loaded_model.score(X_data_test,Y_data_test)


    final_data = type_2_diabetes_data

    X_final = final_data(['Diabetes_012'], axis=1)
    Y_final = final_data[['Diabetes_012']]
    scaler = StandardScaler()
    X_data_test = scaler.fit_transform(X_final)
    Y_data_test = y.to_numpy(Y_final)

    prod_clf = RandomForestClassifier()

    param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

    grid_search = GridSearchCV(prod_clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
    grid_search.fit(X_final,Y_final)

    prod_final_clf = grid_search.best_estimator_

def supportvector():
#Suport vector model code
    type_2_diabetes_data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

    #looking at the headers of the dataset
    type_2_diabetes_data.head(5)

    #looking at all the statistical data from the dataset
    type_2_diabetes_data.describe()

    type_2_diabetes_data.info()

    #heatmap visulisation to see corrilations
    sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

    split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
    for train_indices, test_indices in split.split(type_2_diabetes_data,type_2_diabetes_data[["Diabetes_012","Sex","Age"]]):
        strat_train_set = type_2_diabetes_data.loc[train_indices]
        strat_test_set = type_2_diabetes_data.loc[test_indices]

    #Stratified test set    
    strat_test_set
    #Stratified train set   
    strat_train_set

    strat_train_set.info()


    X_train = strat_train_set.drop(['Diabetes_012'], axis=1)
    y_train = strat_train_set[['Diabetes_012']]
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_train)
    Y_data = y_train.to_numpy()

    X_test = strat_test_set.drop(['Diabetes_012'], axis=1)
    Y_test = strat_test_set[['Diabetes_012']]
    scaler = StandardScaler()
    X_data_test = scaler.fit_transform(X_test)
    Y_data_test = Y_test.to_numpy()

    clf_Svm = svm.SVC(kernel='linear')
    clf_Svm.fit(X_train,y_train)
    
    dump(clf_Svm,filename="clf_random_Support_Vector_model_First.joblib")
    
    X_train_predict = clf_Svm.predict(X_train)
    training_accuracy = accuracy_score(X_train_predict,y_train)
    print('The accuracy of training data is: ',training_accuracy)
    
    
    X_test_predict = clf_Svm.predict(X_test)
    test_accuracy = accuracy_score(X_test_predict,Y_test)
    print('The accuracy of testing data is: ',test_accuracy)
    


""" 
Questionnaire quick understanding key of values before full clean
Diabetes = 1 = no yes = 2
Birthdate = Input from the user 24-12-2024
Gender    =  1 = Male yes = Female 3 = Other       
Family_History = 1 = no yes = 2   
Smoking   = 1 = no yes = 2
Alcohol  = 1 = no yes = 2
Dietry_Habits = 1 = no yes = 2 
Fruit   = 1-5 values ranging from 1 being never to 5 being extremely often
Vegetables  = 1-5 values ranging from 1 being never to 5 being extremely often
Fast_Food  = 1-5 values ranging from 1 being never to 5 being extremely often
Sweets  =  1-5 values ranging from 1 being never to 5 being extremely often
Sleep=  1 = under 4
        2 = 4-6
        3 = 6-8
        4 = 8-10
        5 = 10+
Physical_Activity =  1-5 values ranging from 1 being low to 5 being extremely High 
Energy_Levels  = 1-5 values ranging from 1 being Low to 5 being extremely High
Water    =  1-5 values ranging from 1 being never to 5 being extremely often
Juice  =  1-5 values ranging from 1 being never to 5 being extremely often
Soda    =   1-5 values ranging from 1 being never to 5 being extremely often         
Height   =    Input from the paticipant in cm      
Weight    =   Input from the paticipant in kg      
Waist      =   Input from the paticipant in cm     
Blood_pressure =  Input from the paticipant in 120/90
Glucose    = Input from the paticipant in 5.0
Cholesterol = Input from the paticipant in 4.0                                                      
"""

def calc_age(dob):
    born =datetime.strptime(dob,"%d-%m-%Y").date()
    now = date.today()
    age = now.year - born.year
    return age

def calc_blood_pressure(bp):
    bp = bp.split("/")
    #print(bp)
    if bp[0] == 'nan':
        return None
    else:
        systolic = bp[0]
        diastolic = bp[1]
        if int(systolic) >= 140 or int(diastolic) >= 90:
            result = 1
        else:
            result = 2
    
    return result


def shift_zero_indexing(one_based_system):
    zero_based_system = one_based_system - 1
    return zero_based_system

def shift_zero_indexing_Yes_No(one_based_system):
    if one_based_system == 2:
       zero_based_system = 0
    else:
        zero_based_system = 1  
    return zero_based_system


class Waist_Imputer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Waist'] = imputer.fit_transform(X[['Waist']])
        return X

class Blood_pressure_Imputer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Blood_Pressure'] = imputer.fit_transform(X[['Blood_Pressure']])
        return X

class Glucose_Imputer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Glucose'] = imputer.fit_transform(X[['Glucose']])
        return X

class Cholestrol_Imputer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Cholesterol'] = imputer.fit_transform(X[['Cholesterol']])
        return X

def test(clf,filename):
        test_data_end = pd.read_csv('datasets/questionnaire_train_data.csv')

        #Cleaning the dataset for use e.g cacl age , numbering system change
        #change numbering system to zero based
        
        test_data_end['Gender'] = test_data_end['Gender'].apply(shift_zero_indexing)
        
        test_data_end['Family_History'] = test_data_end['Family_History'].apply(shift_zero_indexing)
        
        test_data_end['Smoking'] = test_data_end['Smoking'].apply(shift_zero_indexing)
        
        test_data_end['Alcohol'] = test_data_end['Alcohol'].apply(shift_zero_indexing)
        
        test_data_end['Dietry_Habits'] = test_data_end['Dietry_Habits'].apply(shift_zero_indexing)
        
        test_data_end['Fruit'] = test_data_end['Fruit'].apply(shift_zero_indexing)
        
        test_data_end['Vegetables'] = test_data_end['Vegetables'].apply(shift_zero_indexing)
        
        test_data_end['Fast_Food'] = test_data_end['Fast_Food'].apply(shift_zero_indexing)
        
        test_data_end['Sweets'] = test_data_end['Sweets'].apply(shift_zero_indexing)
        
        test_data_end['Sleep'] = test_data_end['Sleep'].apply(shift_zero_indexing)
        
        test_data_end['Physical_Activity'] = test_data_end['Physical_Activity'].apply(shift_zero_indexing)
        
        test_data_end['Energy_Levels'] = test_data_end['Energy_Levels'].apply(shift_zero_indexing)
        
        test_data_end['Water'] = test_data_end['Water'].apply(shift_zero_indexing)
        
        test_data_end['Juice'] = test_data_end['Juice'].apply(shift_zero_indexing)
        
        test_data_end['Soda'] = test_data_end['Soda'].apply(shift_zero_indexing)
        #calc values
        test_data_end['Birthdate'] = test_data_end['Birthdate'].apply(calc_age)
        
        #test_data_end['Blood_Pressure'] = test_data_end['Blood_Pressure'].astype(str)
        #test_data_end['Blood_Pressure']
        #test_data_end['Blood_Pressure'] = test_data_end['Blood_Pressure'].apply(calc_blood_pressure)
        
        
        # todo adding the missing values to dataset
        #pipeline for adding the missing values
        """pipeline = Pipeline([("Waist_Imputer",Waist_Imputer()),
                                ("Blood_pressure_Imputer",Blood_pressure_Imputer()),
                                ("Glucose_Imputer",Glucose_Imputer()),
                                ("Cholestrol_Imputer",Cholestrol_Imputer())])"""
                                
        pipeline = Pipeline([("Waist_Imputer",Waist_Imputer())])
            
        test_data_end =pipeline.fit_transform(test_data_end)
            
        #todo create BMI 
        test_data_end['BMI'] = (test_data_end['Weight'] / test_data_end['Height'] / test_data_end['Height']) * 10000
        test_data_end['BMI'] = test_data_end['BMI'].round(1)
            
        #todo create waist/height
        test_data_end['WHtR'] = test_data_end['Waist'] / test_data_end['Height']
        test_data_end['WHtR'] = test_data_end['WHtR'].round(1)
            
            
        #todo create a feature dropper for the glucose-colestroral
        #,"Weight","Height"
        test_data_end = test_data_end.drop(["Diabetes","Glucose","Blood_Pressure","Cholesterol","Weight","Height","Dietry_Habits","Smoking","Alcohol"], axis=1, errors="ignore")
            
        #test_data_end.head(5)
        
        Final_test_data_end = pipeline.fit_transform(test_data_end)
        
        X_test_data_end = Final_test_data_end
        scaler_test_data_end = StandardScaler()
        X_test_data_end = scaler_test_data_end.fit_transform(X_test_data_end)
        
        prediction = clf.predict(X_test_data_end)
        prediction
        
        feature_names_end = test_data_end.columns
        feature_names_end = feature_names_end.tolist()
        
        prediction_weight = show_prediction(clf, X_test_data_end[-1],feature_names = feature_names_end,show_feature_values=True)
        return prediction,prediction_weight


def mydataset_SVM(filename):
    #Suport vector model code
    type_2_diabetes_data = pd.read_csv('datasets/Not_Cleaned_data_project_38184_2024_12_24.csv')

    # todo Clean dataset
    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Gender'] = type_2_diabetes_data['Gender'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Family_History'] = type_2_diabetes_data['Family_History'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Smoking'] = type_2_diabetes_data['Smoking'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Alcohol'] = type_2_diabetes_data['Alcohol'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Dietry_Habits'] = type_2_diabetes_data['Dietry_Habits'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Fruit'] = type_2_diabetes_data['Fruit'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Vegetables'] = type_2_diabetes_data['Vegetables'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Fast_Food'] = type_2_diabetes_data['Fast_Food'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Sweets'] = type_2_diabetes_data['Sweets'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Sleep'] = type_2_diabetes_data['Sleep'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Physical_Activity'] = type_2_diabetes_data['Physical_Activity'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Energy_Levels'] = type_2_diabetes_data['Energy_Levels'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Water'] = type_2_diabetes_data['Water'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Juice'] = type_2_diabetes_data['Juice'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Soda'] = type_2_diabetes_data['Soda'].apply(shift_zero_indexing)
    #calc values
    type_2_diabetes_data['Birthdate'] = type_2_diabetes_data['Birthdate'].apply(calc_age)
    
    #type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].astype(str)
    #type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].apply(calc_blood_pressure)
    
    #type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].astype(int)
    
    # todo adding the missing values to dataset
    #pipeline for adding the missing values
    """pipeline = Pipeline([("Waist_Imputer",Waist_Imputer()),
                         ("Blood_pressure_Imputer",Blood_pressure_Imputer()),
                         ("Glucose_Imputer",Glucose_Imputer()),
                         ("Cholestrol_Imputer",Cholestrol_Imputer())])"""
                         
    pipeline = Pipeline([("Waist_Imputer",Waist_Imputer())])
    
    type_2_diabetes_data =pipeline.fit_transform(type_2_diabetes_data)
    
    #todo create BMI 
    type_2_diabetes_data['BMI'] = (type_2_diabetes_data['Weight'] / type_2_diabetes_data['Height'] / type_2_diabetes_data['Height']) * 10000
    type_2_diabetes_data['BMI'] = type_2_diabetes_data['BMI'].round(1)
    
    #todo create waist/height
    type_2_diabetes_data['WHtR'] = type_2_diabetes_data['Waist'] / type_2_diabetes_data['Height']
    type_2_diabetes_data['WHtR'] = type_2_diabetes_data['WHtR'].round(1)
    
    
    #todo create a feature dropper for the glucose-colestroral
    #,"Weight","Height"
    type_2_diabetes_data = type_2_diabetes_data.drop(["Glucose","Blood_Pressure","Cholesterol","Weight","Height","Dietry_Habits","Smoking","Alcohol"], axis=1, errors="ignore")
    
    

    
    #looking at the headers of the dataset
    type_2_diabetes_data.head(5)

    #looking at all the statistical data from the dataset
    type_2_diabetes_data.describe()

    type_2_diabetes_data.info()

    
    
    #heatmap visulisation to see corrilations
    sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

    split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
    for train_indices, test_indices in split.split(type_2_diabetes_data,type_2_diabetes_data[["Diabetes","Family_History"]]):
        strat_train_set = type_2_diabetes_data.loc[train_indices]
        strat_test_set = type_2_diabetes_data.loc[test_indices]

    #Stratified test set    
    strat_test_set
    #Stratified train set   
    strat_train_set

    strat_train_set.info()


    X_train = strat_train_set.drop(['Diabetes'], axis=1)
    y_train = strat_train_set[['Diabetes']]
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_train)
    Y_data = y_train.to_numpy()

    X_test = strat_test_set.drop(['Diabetes'], axis=1)
    Y_test = strat_test_set[['Diabetes']]
    scaler = StandardScaler()
    X_data_test = scaler.fit_transform(X_test)
    Y_data_test = Y_test.to_numpy()

    clf_Svm = svm.SVC(kernel='linear')
    clf_Svm.fit(X_train,y_train)
    
    #dump(clf_Svm,filename="clf_random_Support_Vector_model_First.joblib")
    
    X_train_predict = clf_Svm.predict(X_train)
    training_accuracy = accuracy_score(X_train_predict,y_train)
    print('The accuracy of training data is: ',training_accuracy)
    
    
    X_test_predict = clf_Svm.predict(X_test)
    test_accuracy = accuracy_score(X_test_predict,Y_test)
    print('The accuracy of testing data is: ',test_accuracy)
    
    test(clf_Svm,filename)
    return








def mydataset_RF():
    type_2_diabetes_data = pd.read_csv('datasets/Not_Cleaned_data_project_38184_2024_12_24.csv')

    # todo Clean dataset
    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Gender'] = type_2_diabetes_data['Gender'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Family_History'] = type_2_diabetes_data['Family_History'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Smoking'] = type_2_diabetes_data['Smoking'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Alcohol'] = type_2_diabetes_data['Alcohol'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Dietry_Habits'] = type_2_diabetes_data['Dietry_Habits'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Fruit'] = type_2_diabetes_data['Fruit'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Vegetables'] = type_2_diabetes_data['Vegetables'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Fast_Food'] = type_2_diabetes_data['Fast_Food'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Sweets'] = type_2_diabetes_data['Sweets'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Sleep'] = type_2_diabetes_data['Sleep'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Physical_Activity'] = type_2_diabetes_data['Physical_Activity'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Energy_Levels'] = type_2_diabetes_data['Energy_Levels'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Water'] = type_2_diabetes_data['Water'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Juice'] = type_2_diabetes_data['Juice'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Soda'] = type_2_diabetes_data['Soda'].apply(shift_zero_indexing)
    #calc values
    type_2_diabetes_data['Birthdate'] = type_2_diabetes_data['Birthdate'].apply(calc_age)
    
    #type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].astype(str)
    #type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].apply(calc_blood_pressure)
    
    #type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].astype(int)
    
    # todo adding the missing values to dataset
    #pipeline for adding the missing values
    """pipeline = Pipeline([("Waist_Imputer",Waist_Imputer()),
                         ("Blood_pressure_Imputer",Blood_pressure_Imputer()),
                         ("Glucose_Imputer",Glucose_Imputer()),
                         ("Cholestrol_Imputer",Cholestrol_Imputer())])"""
                         
    pipeline = Pipeline([("Waist_Imputer",Waist_Imputer())])
    
    type_2_diabetes_data =pipeline.fit_transform(type_2_diabetes_data)
    
    #todo create BMI 
    type_2_diabetes_data['BMI'] = (type_2_diabetes_data['Weight'] / type_2_diabetes_data['Height'] / type_2_diabetes_data['Height']) * 10000
    type_2_diabetes_data['BMI'] = type_2_diabetes_data['BMI'].round(1)
    
    #todo create waist/height
    type_2_diabetes_data['WHtR'] = type_2_diabetes_data['Waist'] / type_2_diabetes_data['Height']
    type_2_diabetes_data['WHtR'] = type_2_diabetes_data['WHtR'].round(1)
    
    
    #todo create a feature dropper for the glucose-colestroral
    #,"Weight","Height"
    type_2_diabetes_data = type_2_diabetes_data.drop(["Glucose","Blood_Pressure","Cholesterol","Weight","Height","Dietry_Habits","Smoking","Alcohol"], axis=1, errors="ignore")
    
    
    # * looking at the headers of the dataset
    #type_2_diabetes_data.head(12)

    # * looking at all the statistical data from the dataset
    #type_2_diabetes_data.describe()

    #type_2_diabetes_data.info()
    
    # ** Since The blood pressure, Glucose and cholestrol have been dropped that means the estimators are no longer needed but will still be kept if used in the future
    # * estimators need to fill in values for waist,blood pressure, Glucose and cholestrol
    # *  Column             Non-Null Count  Dtype  
    """---  ------             --------------  -----  
    0   Diabetes           53 non-null     int64  
    1   Birthdate          53 non-null     int64  
    2   Gender             53 non-null     int64  
    3   Family_History     53 non-null     int64  
    4   Smoking            53 non-null     int64  
    5   Alcohol            53 non-null     int64  
    6   Dietry_Habits      53 non-null     int64  
    7   Fruit              53 non-null     int64  
    8   Vegetables         53 non-null     int64  
    9   Fast_Food          53 non-null     int64  
    10  Sweets             53 non-null     int64  
    11  Sleep              53 non-null     int64  
    12  Physical_Activity  53 non-null     int64  
    13  Energy_Levels      53 non-null     int64  
    14  Water              53 non-null     int64  
    15  Juice              53 non-null     int64  
    16  Soda               53 non-null     int64  
    17  Height             53 non-null     int64  
    18  Weight             53 non-null     int64  
    19  Waist              50 non-null     float64
    20  Blood_Pressure     26 non-null     float64
    21  Glucose            12 non-null     float64
    22  Cholesterol        9 non-null      float64 """
    
    #heatmap visulisation to see corrilations
    #sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

    
   #** Splits the dataset into training and test sets with 20% of the data being reserved for testing
    split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
    for train_indices, test_indices in split.split(type_2_diabetes_data,type_2_diabetes_data[["Diabetes","Family_History"]]):
        strat_train_set = type_2_diabetes_data.loc[train_indices]
        strat_test_set = type_2_diabetes_data.loc[test_indices]

    #Stratified test set    
    strat_test_set
    #Stratified train set   
    strat_train_set

    strat_train_set.info()
    strat_test_set.info()
    
    
    #strat_train_set = pipeline.fit_transform(strat_train_set)
    #strat_train_set.info()


    #** split dataset train model code
    X = strat_train_set.drop(['Diabetes'], axis=1)
    y = strat_train_set[['Diabetes']]
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X)
    Y_data = y.to_numpy()
    #random forest set up
    clf = RandomForestClassifier()

    param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

    grid_search = GridSearchCV(clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
    grid_search.fit(X_data,Y_data)


    #display the CLF

    final_clf = grid_search.best_estimator_


    X_test = strat_test_set.drop(['Diabetes'], axis=1)
    Y_test = strat_test_set[['Diabetes']]
    scaler = StandardScaler()
    X_data_test = scaler.fit_transform(X_test)
    Y_data_test = Y_test.to_numpy()

    # * test the score of the model
    #*final_clf.score(X_data_test,Y_data_test)
    
    feature_names = type_2_diabetes_data.columns
    feature_names = feature_names.delete(0)
    feature_names = feature_names.tolist()
    #* Weights of the model
    #*show_weights(final_clf,feature_names=feature_names)
    

    #exporting file
    #dump(final_clf,filename="clf_random_forest_model_QuestionnaireDataset.joblib")

    #importing file
    #loaded_model = load(filename="clf_random_forest_model_First.joblib")
        
    #joblib_y_preds = loaded_model.predict(X_test)
    #loaded_model.score(X_data_test,Y_data_test)

    # ** Full dataset train model code
    final_data = type_2_diabetes_data

    X_final = final_data.drop(['Diabetes'], axis=1)
    Y_final = final_data[['Diabetes']]
    scaler = StandardScaler()
    X_data_test_final = scaler.fit_transform(X_final)
    Y_data_test_final = Y_final.to_numpy()

    prod_clf = RandomForestClassifier()

    prod_param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

    prod_grid_search = GridSearchCV(prod_clf,prod_param_gird,cv=3,scoring="accuracy",return_train_score=True)
    prod_grid_search.fit(X_data_test_final,Y_data_test_final)

    prod_final_clf = grid_search.best_estimator_
    # * test the score of the model
    #final_clf.score(X_data_test_final,Y_data_test_final)

    feature_names = type_2_diabetes_data.columns
    feature_names = feature_names.delete(0)
    feature_names = feature_names.tolist()
    
    #show_weights(prod_final_clf,feature_names=feature_names)
    
    prediction = prod_final_clf.predict(X_data_test_final)
    prediction
    
    #show_prediction(prod_final_clf, X_data_test_final[3],feature_names = feature_names,show_feature_values=True)
    test(prod_final_clf)
    #todo creat script for the weighting to show risk facotrs 
    
#mydataset_RF()
console = Console()
app = typer.Typer()

#APP 
@app.command("insert")
def questionaire_Create():
    """
    Inserts dataset
    """
    Title = 'Inserts dataset'
    print(pyfiglet.figlet_format(Title))

@app.command("split")
def questionaire_Create():
    """
    Splits Data into training and test data
    """
    Title = 'Split Data'
    print(pyfiglet.figlet_format(Title))

@app.command("Model")
def questionaire_Create():
    """
    Select model for training/
    """
    Title = 'Select model for training'
    print(pyfiglet.figlet_format(Title))

@app.command("train")
def questionaire_Create():
    """
    Begins training the model on dataset
    """
    Title = 'Begin training'
    print(pyfiglet.figlet_format(Title))
    
#have the option to save the model at the end and add the accuracy of it    
@app.command("test")
def questionaire_Create():
    """
    Model makes predictions and will evaluate the accuracy of them
    """
    Title = 'Begin test/'
    print(pyfiglet.figlet_format(Title))
    
    
