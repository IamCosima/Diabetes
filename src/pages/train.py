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
    


def calc_age(dob):
    born =datetime.strptime(dob,"%d-%m-%Y").date()
    now = date.today()
    age = now.year - born.year
    return age

def calc_blood_pressure(bp):
    bp = bp.split('/')
    systolic = bp[0]
    diastolic = bp[1]
    
    if systolic >= 140 or diastolic >= 90:
        result = 1
    else:
        result = 2
    
    return result

""" 
Questionnaire quick understanding key of values before full clean
Diabetes = 1 = no yes = 2
Gender    =  1 = Male yes = Female 3 = Other       
Family-History = 1 = no yes = 2   
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

def shift_zero_indexing(one_based_system):
    zero_based_system = one_based_system - 1
    return

def mydataset():
    type_2_diabetes_data = pd.read_csv('Not_Cleaned_data_project_38184_2024_12_24.csv')

    #Cleaning the dataset for use e.g cacl age , numbering system change
    #change numbering system to zero based
    type_2_diabetes_data['Diabetes'] = type_2_diabetes_data['Diabetes'].apply(shift_zero_indexing)
    
    #calc values
    type_2_diabetes_data['Birthdate'] = type_2_diabetes_data['Birthdate'].apply(calc_age)
    type_2_diabetes_data['Blood_Pressure'] = type_2_diabetes_data['Blood_Pressure'].apply(calc_blood_pressure)



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
    
    
