#import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
#import pyfiglet
from typing_extensions import Annotated
#import inquirer
from datetime import datetime, date 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from joblib import dump,load
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from eli5 import show_weights
from eli5 import show_prediction
from sklearn.inspection import permutation_importance
import time
#from rfpimp import *
    
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



def BMI(bmi):
    """BMI ranges: For adults, 
    BMI ranges are: 
    Underweight: Below 18.5 
    Healthy weight: 18.5 to 24.9 
    Overweight: 25 to 29.9 
    Obese: 30 to 39.9 
    Severely obese: 40 or above"""
    
    if bmi <= 18.5:
        result = "Underweight\n"
    elif bmi >= 18.5 and bmi <= 24.9 :
        result = "Healthy weight\n"
    elif bmi >= 25.0 and bmi <= 29.9 :
        result = "Overweight\n"
    elif bmi >= 30.0 and bmi <= 39.9 :
        result = "Obese\n"
    elif bmi >= 40:
        result = "Severely obese\n"
    else:
        result = "Error\n"
    return result

def WHtR(whrt):
    """WHtR ranges:
    WHtR ranges are: 
    Healthy: 0.4 to 0.49
    Increased: 0.5 to 0.59
    High: 0.6+
    """
    if whrt >= 0.4 and whrt <= 0.49:
        result = "Healthy\n"
    elif whrt >= 0.5  and whrt <= 0.59 :
        result = "Increased risk of health complications\n"
    elif whrt >= 0.6:
        result = "High risk of health complications\n"
    else:
        result = "Error\n"
    return result

def Age(age):
    """Age ranges:
    Healthy: under 21 low risk
    Increased: 22 - 44 moderate risk
    High: 45+ High risk
    """
    if age <= 21:
        result = "low risk risk of health complications\n"
    elif age >= 22  and age <= 44 :
        result = "Increased risk of health complications\n"
    elif age >= 45:
        result = "High risk of health complications\n"
    else:
        result = "Error\n"
    return result
    
# def plot_feature_importances(perm_importance_result, feat_name):
#     """bar plot the feature importance"""

#     fig, ax = plt.subplots()

#     indices = perm_importance_result["importances_mean"].argsort()
#     plt.barh(
#         range(len(indices)),
#         perm_importance_result["importances_mean"][indices],
#         xerr=perm_importance_result["importances_std"][indices],
#     )

#     ax.set_yticks(range(len(indices)))
#     _ = ax.set_yticklabels(feat_name[indices])


def test(clf,filename):
        #test_data_end = pd.read_csv('datasets/questionnaire_user_data.csv')
    test_data_end = pd.read_csv(filename)
        #Cleaning the dataset for use e.g cacl age , numbering system change
        #change numbering system to zero based
        
    test_data_end['Gender'] = test_data_end['Gender'].apply(shift_zero_indexing)
        
    test_data_end['Family_History'] = test_data_end['Family_History'].apply(shift_zero_indexing_Yes_No)
        
    test_data_end['Smoking'] = test_data_end['Smoking'].apply(shift_zero_indexing_Yes_No)
        
    test_data_end['Alcohol'] = test_data_end['Alcohol'].apply(shift_zero_indexing_Yes_No)
        
    test_data_end['Dietry_Habits'] = test_data_end['Dietry_Habits'].apply(shift_zero_indexing_Yes_No)
        
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
    test_data_end = test_data_end.drop(["Glucose","Blood_Pressure","Cholesterol","Weight","Height","Dietry_Habits","Smoking","Alcohol"], axis=1, errors="ignore")
            
        #print(test_data_end.head(-1))
        
    Final_test_data_end = test_data_end
        
    X_test_data_end = Final_test_data_end
    #scaler_test_data_end = StandardScaler()
    #X_test_data_end = scaler_test_data_end.fit_transform(X_test_data_end)
    normalize = Normalizer()
    X_test_data_end = normalize.fit_transform(X_test_data_end)
        
    prediction_result= clf.predict(X_test_data_end)
    print(prediction_result)
        
    feature_names_end = test_data_end.columns
    feature_names_end = feature_names_end.tolist()
        
    #prediction_weight = show_prediction(clf, X_test_data_end[-1],feature_names = feature_names_end,show_feature_values=True)
        
        #todo creat script for the weighting to show risk facotrs 
        
    bmi = test_data_end['BMI'].tolist()
    bmi = bmi[-1]
        
    waistcir = test_data_end['WHtR'].tolist()
    waistcir = waistcir[-1]
        
    age = test_data_end['Birthdate'].tolist()
    age = age[-1]
        
    sleep = test_data_end['Sleep'].tolist()
    sleep = sleep[-1]
        
    activity = test_data_end['Physical_Activity'].tolist() 
    activity = activity[-1]
        
    energy = test_data_end['Energy_Levels'].tolist()
    energy = energy[-1]
        
    fruit = test_data_end['Fruit'].tolist()
    fruit = fruit[-1]
        
    veg = test_data_end['Vegetables'].tolist() 
    veg = veg[-1]
        
    fast = test_data_end['Fast_Food'].tolist()
    fast = fast[-1]
        
    sweets = test_data_end['Sweets'].tolist()
    sweets = sweets[-1]
        
    water = test_data_end['Water'].tolist()
    water = water[-1]
        
    juice =test_data_end['Juice'].tolist()
    juice = juice[-1]
        
    soda = test_data_end['Soda'].tolist()
    soda = soda[-1]

    if prediction_result[-1] == 0:
        disclaimer = "* Disclaimer this is not a medical diagnosis Please Consult Your Doctor to verify\n"
        script = "You have a low risk of Type 2 Diabetes\n"
        script_BMI = "Your BMI is " + str(bmi) +" which means that you are " + BMI(bmi)
        script_WHtR = "Your Waist-Height Ratio is " + str(waistcir) +" which means that your are " + WHtR(waistcir)
        #script_Age = "Your Age is " + str(age) + " which means that you are at a " + Age(age)
        script = disclaimer + script + script_BMI + script_WHtR
    else:
        disclaimer = "*Disclaimer this is not a medical diagnosis Please Consult Your Doctor to verify\n"
        script = "You have a High Risk of Type 2 Diabetes\n"
        script_BMI = "Your BMI is " + str(bmi) +" which means that you are " + BMI(bmi)
        script_WHtR = "Your Waist-Height Ratio is " + str(waistcir) +" which means that your are " + WHtR(waistcir)
        #script_Age = "Your Age is " + str(age) + " which means that you are at a " + Age(age)
        script = disclaimer + script+ script_BMI + script_WHtR
    
    
    return print(script)


def mydataset_SVM():
    #Suport vector model code
    type_2_diabetes_data = pd.read_csv('datasets/Not_Cleaned_data_project_38184_2024_12_24.csv')
    #type_2_diabetes_data = pd.read_csv(filename)
    # todo Clean dataset
    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Gender'] = type_2_diabetes_data['Gender'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Family_History'] = type_2_diabetes_data['Family_History'].apply(shift_zero_indexing_Yes_No)
    
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
    #type_2_diabetes_data.head(5)

    #looking at all the statistical data from the dataset
    #type_2_diabetes_data.describe()

    #type_2_diabetes_data.info()

    
    
    #heatmap visulisation to see corrilations
    #sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

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
    #scaler = StandardScaler()
    #X_data = scaler.fit_transform(X_train)
    normalize = Normalizer()
    X_data = normalize.fit_transform(X_train)
    Y_data = y_train.to_numpy()

    X_test = strat_test_set.drop(['Diabetes'], axis=1)
    Y_test = strat_test_set[['Diabetes']]
    #scaler = StandardScaler()
    #X_data_test = scaler.fit_transform(X_test)
    normalize = Normalizer()
    X_data_test = normalize.fit_transform(X_test)
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
    
    filename = "datasets/Not_Cleaned_data_project_38184_2024_12_24.csv"
    test(clf_Svm,filename)
    return

def mydataset_RF():
    type_2_diabetes_data = pd.read_csv('datasets/Not_Cleaned_data_project_38184_2024_12_24.csv')
    #type_2_diabetes_data = pd.read_csv(filename)
    # todo Clean dataset
    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Gender'] = type_2_diabetes_data['Gender'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Family_History'] = type_2_diabetes_data['Family_History'].apply(shift_zero_indexing_Yes_No)
    
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
    #type_2_diabetes_data.head(90)

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
    22  Cholesterol        9 non-null      float64 
    
    Data columns (total 17 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Diabetes           90 non-null     int64  
 1   Birthdate          90 non-null     int64  
 2   Gender             90 non-null     int64  
 3   Family_History     90 non-null     int64  
 4   Fruit              90 non-null     int64  
 5   Vegetables         90 non-null     int64  
 6   Fast_Food          90 non-null     int64  
 7   Sweets             90 non-null     int64  
 8   Sleep              90 non-null     int64  
 9   Physical_Activity  90 non-null     int64  
 10  Energy_Levels      90 non-null     int64  
 11  Water              90 non-null     int64  
 12  Juice              90 non-null     int64  
 13  Soda               90 non-null     int64  
 14  Waist              90 non-null     float64
 15  BMI                90 non-null     float64
 16  WHtR               90 non-null     float64
    """
    
    #heatmap visulisation to see corrilations
    #cff7d2
    #295b3e
    
    
    #*Distribution type graphs
    plt.figure()
    label = ['Non-Diabetic','Type 2 Diabetic']
    dia_counts = type_2_diabetes_data['Diabetes'].value_counts(dropna=False)
    plt.title('Distribution of Participant diabetic status')
    plt.pie(dia_counts, labels=label,autopct='%.0f%%',colors=sns.color_palette("pastel", 8),) 
    plt.show()
        
    #dia = sns.displot(data=type_2_diabetes_data, x='Diabetes', bins = 2,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    #dia.set(title= "Distribution of People With Type 2 diabetes",xlabel = "Diabetes: 0 = non-diabetic, 1 = diabetic",xmargin=0.1 ,xlim =(0,1), xticks=(0,1))
    
    birth = sns.displot(data=type_2_diabetes_data, x='Birthdate',bins = 20,binwidth=10,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    birth.set(title = "Distribution of Age",xmargin=0)
    
    gender = sns.displot(data=type_2_diabetes_data, x='Gender',bins = 2,binwidth = 0.6,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    gender.set(title = "Distribution of Gender",xmargin=0.1 ,xlim =(0,2), xticks=(0,2))
    
    Family_History = sns.displot(data=type_2_diabetes_data, x='Family_History',bins = 2,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Family_History.set(title = "Distribution of Family_History",xmargin=0.1 ,xlim =(0,1), xticks=(0,1))
    
    Fruit = sns.displot(data=type_2_diabetes_data, x='Fruit',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Fruit.set(title = "Distribution of Fruit Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Vegetables = sns.displot(data=type_2_diabetes_data, x='Vegetables',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Vegetables.set(title = "Distribution of Vegetables Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Fast_Food = sns.displot(data=type_2_diabetes_data, x='Fast_Food',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Fast_Food.set(title = "Distribution of Fast Food",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Sweets = sns.displot(data=type_2_diabetes_data, x='Sweets',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Sweets.set(title = "Distribution of Sweets Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Sleep =  sns.displot(data=type_2_diabetes_data, x='Sleep',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Sleep.set(title = "Distribution of Sleep Duration",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Physical_Activity = sns.displot(data=type_2_diabetes_data, x='Physical_Activity',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Physical_Activity.set(title = "Distribution of Physical Activity",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Energy_Levels = sns.displot(data=type_2_diabetes_data, x='Energy_Levels',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Energy_Levels.set(title = "Distribution of Energy Levels",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Water = sns.displot(data=type_2_diabetes_data, x='Water',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Water.set(title = "Distribution of Water Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Juice = sns.displot(data=type_2_diabetes_data, x='Juice',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Juice.set(title = "Distribution of Juice Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    Soda = sns.displot(data=type_2_diabetes_data, x='Soda',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    Soda.set(title = "Distribution of Soda Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    
    BMI = sns.displot(data=type_2_diabetes_data, x='BMI',bins = 5,color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    BMI.set(title = "Distribution of BMI ",xmargin=0.1)
    
    
    WHtR = sns.displot(data=type_2_diabetes_data, x='WHtR',color="#cff7d2",edgecolor='#295b3e',linewidth = 1) 
    WHtR.set(title = "Distribution of WHtR ",xmargin=0.1,xlim =(0,1))
    
    '''
        Data columns (total 17 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Diabetes           90 non-null     int64  
 1   Birthdate          90 non-null     int64  
 2   Gender             90 non-null     int64  
 3   Family_History     90 non-null     int64  
 4   Fruit              90 non-null     int64  
 5   Vegetables         90 non-null     int64  
 6   Fast_Food          90 non-null     int64  
 7   Sweets             90 non-null     int64  
 8   Sleep              90 non-null     int64  
 9   Physical_Activity  90 non-null     int64  
 10  Energy_Levels      90 non-null     int64  
 11  Water              90 non-null     int64  
 12  Juice              90 non-null     int64  
 13  Soda               90 non-null     int64  
 14  Waist              90 non-null     float64
 15  BMI                90 non-null     float64
 16  WHtR               90 non-null     float64
    '''
    
    
    #* Relationship type graphs
    
    # diet = type_2_diabetes_data[['Birthdate','Fruit','Vegetables','Fast_Food','Sweets','Water','Juice','Soda']]
    # diet_data = pd.melt(diet,id_vars=['Birthdate'])
    # sns.lineplot(data=diet_data,x= 'Birthdate',y='value', hue='variable',estimator=None)
    # sns.set(title = "Distribution of Soda Consumption",xmargin=0.1 ,xlim =(0,4), xticks=(0,4))
    
    corrheat = sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")
    
    BMI_WHR  = sns.lineplot(data = type_2_diabetes_data, x="BMI" ,y = 'WHtR',color="#cff7d2",)
    BMI_WHR.set(title = "Relationship between BMI vs WHtR  ",xmargin=0.1)
    
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
    #scaler = StandardScaler()
    #X_data = scaler.fit_transform(X)
    normalize = Normalizer()
    X_data = normalize.fit_transform(X)
    Y_data = y.to_numpy()
    #random forest set up
    clf = RandomForestClassifier()

    #param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]
    param_gird = [{ "n_estimators": [50, 100, 150], "max_depth": [None, 10, 20],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4],"max_features": ['sqrt', 'log2', None]}]
    
    
    
    grid_search = GridSearchCV(clf,param_gird,cv=3,scoring="accuracy",verbose=1,return_train_score=True)
    grid_search.fit(X_data,Y_data.ravel())


    #display the CLF

    final_clf = grid_search.best_estimator_


    X_test = strat_test_set.drop(['Diabetes'], axis=1)
    Y_test = strat_test_set[['Diabetes']]
    #scaler = StandardScaler()
    #X_data_test = scaler.fit_transform(X_test)
    normalize = Normalizer()
    X_data_test = normalize.fit_transform(X_test)
    
    Y_data_test = Y_test.to_numpy()

    # * test the score of the model
    final_clf.score(X_data_test,Y_data_test)
    
    feature_names = type_2_diabetes_data.columns
    feature_names = feature_names.delete(0)
    feature_names = feature_names.tolist()
    #* Weights of the model
    show_weights(final_clf,feature_names=feature_names)
    

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
    #scaler = StandardScaler()
    #X_data_test_final = scaler.fit_transform(X_final)
    normalize = Normalizer()
    X_data_test_final = normalize.fit_transform(X_final)
    
    Y_data_test_final = Y_final.to_numpy()

    prod_clf = RandomForestClassifier()

    #prod_param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]
    prod_param_gird = [{ "n_estimators": [50, 100, 150], "max_depth": [None, 10, 20],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4],"max_features": ['sqrt', 'log2', None]}]
    
    prod_grid_search = GridSearchCV(prod_clf,prod_param_gird,cv=5,scoring="accuracy",verbose=1,return_train_score=True)
    prod_grid_search.fit(X_data_test_final,Y_data_test_final.ravel())

    prod_final_clf = prod_grid_search.best_estimator_
    # * test the score of the model
    prod_final_clf.score(X_data_test_final,Y_data_test_final)

    feature_names = type_2_diabetes_data.columns
    feature_names = feature_names.delete(0)
    feature_names = feature_names.tolist()
    
    #show_weights(prod_final_clf,feature_names=feature_names)
    
    #Fix = final_data.drop(['Diabetes'],axis=1)
    #fixing = scaler.fit_transform(Fix)
    #prediction = prod_final_clf.predict(fixing)
   
    prediction_Prod = prod_final_clf.predict(X_data_test_final)
    
    prediction_test = final_clf.predict(X_test)
    
    
    conf_matrix_Stratified = confusion_matrix(Y_data_test, prediction_test)
    confusion_matrix_strat = sns.heatmap(conf_matrix_Stratified, annot=True, fmt='d',cmap="viridis" )
    # set x-axis label and ticks. 
    confusion_matrix_strat.set_xlabel("Predicted Diagnosis")
    confusion_matrix_strat.xaxis.set_ticklabels(['Negative', 'Positive'])
 
    # set y-axis label and ticks
    confusion_matrix_strat.set_ylabel("Actual Diagnosis")
    confusion_matrix_strat.yaxis.set_ticklabels(['Negative', 'Positive'])
    
    
    
    conf_matrix_prod = confusion_matrix(Y_data_test_final, prediction_Prod)
    confusion_matrix_prod =sns.heatmap(conf_matrix_prod, annot=True, fmt='d',cmap="viridis" )
     # set x-axis label and ticks. 
    confusion_matrix_prod.set_xlabel("Predicted Diagnosis")
    confusion_matrix_prod.xaxis.set_ticklabels(['Negative', 'Positive'])
 
    # set y-axis label and ticks
    confusion_matrix_prod.set_ylabel("Actual Diagnosis")
    confusion_matrix_prod.yaxis.set_ticklabels(['Negative', 'Positive'])
    
   
    
    #show_prediction(prod_final_clf, X_data_test_final[3],,feature_names = feature_names,show_feature_values=True)
    filename = "datasets/Not_Cleaned_data_project_38184_2024_12_24.csv"
    test(prod_final_clf,filename)
   

def mydataset_SVM_Prediction(filename):
    #Suport vector model code
    type_2_diabetes_data = pd.read_csv('datasets/Not_Cleaned_data_project_38184_2024_12_24.csv')
    #type_2_diabetes_data = pd.read_csv(filename)
    # todo Clean dataset
    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Gender'] = type_2_diabetes_data['Gender'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Family_History'] = type_2_diabetes_data['Family_History'].apply(shift_zero_indexing_Yes_No)
    
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
    
    final_data = type_2_diabetes_data

    X_final = final_data.drop(['Diabetes'], axis=1)
    Y_final = final_data[['Diabetes']]
    #scaler = StandardScaler()
    #X_data_test_final = scaler.fit_transform(X_final)
    normalize = Normalizer()
    X_data_test_final = normalize.fit_transform(X_final)
    Y_data_test_final = Y_final.to_numpy()
    
    feature_names = type_2_diabetes_data.columns
    feature_names = feature_names.delete(0)
    feature_names = feature_names.tolist()
    
    
    clf_Svm = svm.SVC(kernel='linear',verbose=True)
    clf_Svm.fit(X_data_test_final,Y_data_test_final.ravel())
    
   
    #show_weights(clf_Svm,feature_names=feature_names)
    
    clf = clf_Svm
    prediction = clf.predict(X_data_test_final)
    prediction    
    #prediction_weight = show_prediction(clf_Svm,X_data_test_final[3],feature_names = feature_names_end,show_feature_values=True)
        
    
    
    test(clf_Svm,filename)
    

def mydataset_RF_Prediction(filename):
    type_2_diabetes_data = pd.read_csv('datasets/Not_Cleaned_data_project_38184_2024_12_24.csv')
    #type_2_diabetes_data = pd.read_csv(filename)
    # todo Clean dataset
    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing_Yes_No)
    
    type_2_diabetes_data['Gender'] = type_2_diabetes_data['Gender'].apply(shift_zero_indexing)
    
    type_2_diabetes_data['Family_History'] = type_2_diabetes_data['Family_History'].apply(shift_zero_indexing_Yes_No)
    
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
    
    # ** Full dataset train model code
    final_data = type_2_diabetes_data

    X_final = final_data.drop(['Diabetes'], axis=1)
    Y_final = final_data[['Diabetes']]
    #scaler = StandardScaler()
    #X_data_test_final = scaler.fit_transform(X_final)
    normalize = Normalizer()
    X_data_test_final = normalize.fit_transform(X_final)
    
    Y_data_test_final = Y_final.to_numpy()

    prod_clf = RandomForestClassifier()

    prod_param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

    prod_grid_search = GridSearchCV(prod_clf,prod_param_gird,cv=3,scoring="accuracy",verbose=1,return_train_score=True)
    prod_grid_search.fit(X_data_test_final,Y_data_test_final.ravel())

    prod_final_clf = prod_grid_search.best_estimator_
    # * test the score of the model
    #final_clf.score(X_data_test_final,Y_data_test_final)

    feature_names = type_2_diabetes_data.columns
    feature_names = feature_names.delete(0)
    feature_names = feature_names.tolist()
    
    #show_weights(prod_final_clf,feature_names=feature_names)
    
    #best_features = prod_final_clf.feature_importances_
    
    #feature_names = type_2_diabetes_data.columns
    
    #imp = importances(prod_final_clf,X_final,Y_final)
    #viz = plot_importances(imp)
    #viz.view()
    #importance  = permutation_importance(prod_final_clf,X_data_test_final,Y_data_test_final)
    #plot_feature_importances(importance, feature_names)
  
    #prediction = prod_final_clf.predict(X_data_test_final)
    #prediction
    
    #show_prediction(prod_final_clf, X_data_test_final[2],feature_names = feature_names,show_feature_values=True)
    
    # clf = prod_final_clf
    # test_data_end = pd.read_csv('datasets/questionnaire_user_data.csv')
    #     #test_data_end = pd.read_csv(filename)
    #     #Cleaning the dataset for use e.g cacl age , numbering system change
    #     #change numbering system to zero based
        
    # test_data_end['Gender'] = test_data_end['Gender'].apply(shift_zero_indexing)
        
    # test_data_end['Family_History'] = test_data_end['Family_History'].apply(shift_zero_indexing_Yes_No)
        
    # test_data_end['Smoking'] = test_data_end['Smoking'].apply(shift_zero_indexing_Yes_No)
        
    # test_data_end['Alcohol'] = test_data_end['Alcohol'].apply(shift_zero_indexing_Yes_No)
        
    # test_data_end['Dietry_Habits'] = test_data_end['Dietry_Habits'].apply(shift_zero_indexing_Yes_No)
        
    # test_data_end['Fruit'] = test_data_end['Fruit'].apply(shift_zero_indexing)
        
    # test_data_end['Vegetables'] = test_data_end['Vegetables'].apply(shift_zero_indexing)
        
    # test_data_end['Fast_Food'] = test_data_end['Fast_Food'].apply(shift_zero_indexing)
        
    # test_data_end['Sweets'] = test_data_end['Sweets'].apply(shift_zero_indexing)
        
    # test_data_end['Sleep'] = test_data_end['Sleep'].apply(shift_zero_indexing)
        
    # test_data_end['Physical_Activity'] = test_data_end['Physical_Activity'].apply(shift_zero_indexing)
        
    # test_data_end['Energy_Levels'] = test_data_end['Energy_Levels'].apply(shift_zero_indexing)
        
    # test_data_end['Water'] = test_data_end['Water'].apply(shift_zero_indexing)
        
    # test_data_end['Juice'] = test_data_end['Juice'].apply(shift_zero_indexing)
        
    # test_data_end['Soda'] = test_data_end['Soda'].apply(shift_zero_indexing)
    #     #calc values
    # test_data_end['Birthdate'] = test_data_end['Birthdate'].apply(calc_age)
        
    #     #test_data_end['Blood_Pressure'] = test_data_end['Blood_Pressure'].astype(str)
    #     #test_data_end['Blood_Pressure']
    #     #test_data_end['Blood_Pressure'] = test_data_end['Blood_Pressure'].apply(calc_blood_pressure)
        
        
    #     # todo adding the missing values to dataset                        
    # pipeline = Pipeline([("Waist_Imputer",Waist_Imputer())])
            
    # test_data_end =pipeline.fit_transform(test_data_end)
            
    #     #todo create BMI 
    # test_data_end['BMI'] = (test_data_end['Weight'] / test_data_end['Height'] / test_data_end['Height']) * 10000
    # test_data_end['BMI'] = test_data_end['BMI'].round(1)
            
    #     #todo create waist/height
    # test_data_end['WHtR'] = test_data_end['Waist'] / test_data_end['Height']
    # test_data_end['WHtR'] = test_data_end['WHtR'].round(1)
            
            
    #     #todo create a feature dropper for the glucose-colestroral
    #     #,"Weight","Height"
    # test_data_end = test_data_end.drop(["Glucose","Blood_Pressure","Cholesterol","Weight","Height","Dietry_Habits","Smoking","Alcohol"], axis=1, errors="ignore")
            
    #     #print(test_data_end.head(-1))
        
    # Final_test_data_end = test_data_end
        
    # X_test_data_end = Final_test_data_end
    # #scaler_test_data_end = StandardScaler()
    # #X_test_data_end = scaler_test_data_end.fit_transform(X_test_data_end)
    # normalize = Normalizer()
    # X_test_data_end = normalize.fit_transform(X_test_data_end)
        
    # prediction_result= clf.predict(X_test_data_end)
    # print(prediction_result)
    test(prod_final_clf,filename)

  
  
  
  
  

# def randomforest():
#     type_2_diabetes_data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

#     #looking at the headers of the dataset
#     type_2_diabetes_data.head(5)

#     #looking at all the statistical data from the dataset
#     type_2_diabetes_data.describe()

#     type_2_diabetes_data.info()

#     #heatmap visulisation to see corrilations
#     sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

#     split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
#     for train_indices, test_indices in split.split(type_2_diabetes_data,type_2_diabetes_data[["Diabetes_012","Sex","Age"]]):
#         strat_train_set = type_2_diabetes_data.loc[train_indices]
#         strat_test_set = type_2_diabetes_data.loc[test_indices]

#     #Stratified test set    
#     strat_test_set
#     #Stratified train set   
#     strat_train_set

#     strat_train_set.info()


#     X = strat_train_set.drop(['Diabetes_012'], axis=1)
#     y = strat_train_set[['Diabetes_012']]
#     scaler = StandardScaler()
#     X_data = scaler.fit_transform(X)
#     Y_data = y.to_numpy()
#     #random forest set up
#     clf = RandomForestClassifier()

#     param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

#     grid_search = GridSearchCV(clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
#     grid_search.fit(X_data,Y_data)

#     final_clf = grid_search.best_estimator_


#     X_test = strat_test_set.drop(['Diabetes_012'], axis=1)
#     Y_test = strat_test_set[['Diabetes_012']]
#     scaler = StandardScaler()
#     X_data_test = scaler.fit_transform(X_test)
#     Y_data_test = Y_test.to_numpy()

#     final_clf.score(X_data_test,Y_data_test)

#     #exporting file
#     dump(final_clf,filename="clf_random_forest_model_First.joblib")

#     #importing file
#     loaded_model = load(filename="clf_random_forest_model_First.joblib")\
        
#     joblib_y_preds = loaded_model.predict(X_test)
#     loaded_model.score(X_data_test,Y_data_test)


#     final_data = type_2_diabetes_data

#     X_final = final_data(['Diabetes_012'], axis=1)
#     Y_final = final_data[['Diabetes_012']]
#     scaler = StandardScaler()
#     X_data_test = scaler.fit_transform(X_final)
#     Y_data_test = y.to_numpy(Y_final)

#     prod_clf = RandomForestClassifier()

#     param_gird = [{"n_estimators": [10,100,200,500],"max_depth": [None,5,10],"min_samples_split": [2,3,4]}]

#     grid_search = GridSearchCV(prod_clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
#     grid_search.fit(X_final,Y_final)

#     prod_final_clf = grid_search.best_estimator_

# def supportvector():
# #Suport vector model code
#     type_2_diabetes_data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

#     #looking at the headers of the dataset
#     type_2_diabetes_data.head(5)

#     #looking at all the statistical data from the dataset
#     type_2_diabetes_data.describe()

#     type_2_diabetes_data.info()

#     #heatmap visulisation to see corrilations
#     sns.heatmap(type_2_diabetes_data.corr(), cmap="YlGnBu")

#     split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
#     for train_indices, test_indices in split.split(type_2_diabetes_data,type_2_diabetes_data[["Diabetes_012","Sex","Age"]]):
#         strat_train_set = type_2_diabetes_data.loc[train_indices]
#         strat_test_set = type_2_diabetes_data.loc[test_indices]

#     #Stratified test set    
#     strat_test_set
#     #Stratified train set   
#     strat_train_set

#     strat_train_set.info()


#     X_train = strat_train_set.drop(['Diabetes_012'], axis=1)
#     y_train = strat_train_set[['Diabetes_012']]
#     scaler = StandardScaler()
#     X_data = scaler.fit_transform(X_train)
#     Y_data = y_train.to_numpy()

#     X_test = strat_test_set.drop(['Diabetes_012'], axis=1)
#     Y_test = strat_test_set[['Diabetes_012']]
#     scaler = StandardScaler()
#     X_data_test = scaler.fit_transform(X_test)
#     Y_data_test = Y_test.to_numpy()

#     clf_Svm = svm.SVC(kernel='linear')
#     clf_Svm.fit(X_train,y_train)
    
#     dump(clf_Svm,filename="clf_random_Support_Vector_model_First.joblib")
    
#     X_train_predict = clf_Svm.predict(X_train)
#     training_accuracy = accuracy_score(X_train_predict,y_train)
#     print('The accuracy of training data is: ',training_accuracy)
    
    
#     X_test_predict = clf_Svm.predict(X_test)
#     test_accuracy = accuracy_score(X_test_predict,Y_test)
#     print('The accuracy of testing data is: ',test_accuracy)

# #mydataset_RF()
# console = Console()
# app = typer.Typer()

# #APP 
# @app.command("insert")
# def questionaire_Create():
#     """
#     Inserts dataset
#     """
#     Title = 'Inserts dataset'
#     print(pyfiglet.figlet_format(Title))

# @app.command("split")
# def questionaire_Create():
#     """
#     Splits Data into training and test data
#     """
#     Title = 'Split Data'
#     print(pyfiglet.figlet_format(Title))

# @app.command("Model")
# def questionaire_Create():
#     """
#     Select model for training/
#     """
#     Title = 'Select model for training'
#     print(pyfiglet.figlet_format(Title))

# @app.command("train")
# def questionaire_Create():
#     """
#     Begins training the model on dataset
#     """
#     Title = 'Begin training'
#     print(pyfiglet.figlet_format(Title))
    
# #have the option to save the model at the end and add the accuracy of it    
# @app.command("test")
# def questionaire_Create():
#     """
#     Model makes predictions and will evaluate the accuracy of them
#     """
#     Title = 'Begin test/'
#     print(pyfiglet.figlet_format(Title))
    
    
