import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
import pyfiglet
from typing_extensions import Annotated
import inquirer
import datetime

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump,load

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
    
    
