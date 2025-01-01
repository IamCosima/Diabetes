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
import os
import csv
import train as predict
console = Console()
app = typer.Typer()

Gender = [
    inquirer.List(
        "Gender",
        message="What is your gender?",
        choices=["Male", "Female","Other"],
    ),
]   
Diet = [
    inquirer.List(
        "Diet",
        message="Would your dietary habits be considered as 🍏 Healthy or 🍔 Unhealthy?",
        choices=["Healthy", "Unhealthy"],
    ),
] 
Sleep = [
    inquirer.List(
        "Sleep",
        message="How many ⏰ hours of 💤 sleep do you get a day?",
        choices=["Under 4 hours", "4-6 hours","6-8 hours","8-10 hours","10+"],
    ),
] 

Model = [
    inquirer.List(
        "Model",
        message="Which model would you like to choose?",
        choices=["Decision trees", "Random forests"],
    ),
] 


"""#   Column             Non-Null Count  Dtype  *Generated
---  ------             --------------  -----  
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
 19  Waist              53 non-null     float64
 20  Blood_Pressure     26 non-null     object 
 21  Glucose            12 non-null     float64
 22  Cholesterol        9 non-null      float64
 *23  BMI                53 non-null     float64
 *24  WHtR               53 non-null     float64
    """

def gender_to_numerical(gender):
    if gender.get("Gender") == "Male":
        numerical = 1
    elif gender.get("Gender") == "Female":
        numerical = 2
    else:
        numerical = 3
    return numerical

def Diet_to_numerical(diet):
    if diet.get("Diet") == "Healthy":
        numerical = 1
    elif diet.get("Diet") == "Unhealthy":
        numerical = 2
    else:
        numerical = 3
    return numerical

def Sleep_to_numerical(sleep): 
    choices=["Under 4 hours", "4-6 hours","6-8 hours","8-10 hours","10+"],
    if sleep.get("Sleep") == "Under 4 hours":
        numerical = 1
    elif sleep.get("Sleep") == "4-6 hours":
        numerical = 2
    elif sleep.get("Sleep") == "6-8 hours":
        numerical = 3
    elif sleep.get("Sleep") == "8-10 hours":
        numerical = 4
    else:
        numerical = 5
    return numerical

def confirm_to_numerical(zero_based_system):
    if zero_based_system == 0:
       zero_based_system = 2
    else:
        zero_based_system = 1  
    return zero_based_system
    
    
    
def create_csv_train(Diabetes,birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist):
    csv_headers = "Diabetes,Birthdate,Gender,Family_History,Smoking,Alcohol,Dietry_Habits,Fruit,Vegetables,Fast_Food,Sweets,Sleep,Physical_Activity,Energy_Levels,Water,Juice,Soda,Height,Weight,Waist,Blood_Pressure,Glucose,Cholesterol"
    
    Diabetes = Diabetes
    birthdate = birthdate
    gender = gender
    family = family
    smoker = smoker
    alcohol = alcohol
    diet = diet
    fruit = fruit
    vegtables = vegtables
    fast_food = fast_food
    sweets = sweets
    sleep =sleep
    activity = activity
    energy_levels = energy_levels
    water = water
    juice = juice
    soda = soda
    height = height
    weight = weight
    waist = waist
    blood_pressure = 0
    glucose = 0
    Cholesterol = 0
    filename = "datasets/questionnaire_train_data.csv"
    
    
    questionnaire_data_header_inc = [
                        ["Diabetes","Birthdate","Gender","Family_History","Smoking","Alcohol","Dietry_Habits","Fruit","Vegetables","Fast_Food","Sweets","Sleep","Physical_Activity","Energy_Levels","Water","Juice","Soda","Height","Weight","Waist","Blood_Pressure","Glucose","Cholesterol"],
                        [Diabetes,birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist,blood_pressure,glucose,Cholesterol]
                        ]
    questionnaire_data = [[Diabetes,birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist,blood_pressure,glucose,Cholesterol]]
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
        
        with open(filename, append_write, newline='') as csvfile:
            questionnaire = csv.writer(csvfile)
            questionnaire.writerows(questionnaire_data)
    else:
        append_write = 'w' # make a new file if not
        with open(filename, append_write, newline='') as csvfile:
            questionnaire = csv.writer(csvfile)
            questionnaire.writerows(questionnaire_data_header_inc)
            
            
    return filename
    
        
    
    
 

def create_csv_test(birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist):
    csv_headers = "Birthdate,Gender,Family_History,Smoking,Alcohol,Dietry_Habits,Fruit,Vegetables,Fast_Food,Sweets,Sleep,Physical_Activity,Energy_Levels,Water,Juice,Soda,Height,Weight,Waist,Blood_Pressure,Glucose,Cholesterol"
    birthdate = birthdate
    gender = gender
    family = family
    smoker = smoker
    alcohol = alcohol
    diet = diet
    fruit = fruit
    vegtables = vegtables
    fast_food = fast_food
    sweets = sweets
    sleep =sleep
    activity = activity
    energy_levels = energy_levels
    water = water
    juice = juice
    soda = soda
    height = height
    weight = weight
    waist = waist
    blood_pressure = 0
    glucose = 0
    Cholesterol = 0
    filename = "datasets/questionnaire_user_data.csv"
    
    
    questionnaire_data_header_inc = [
                        ["Birthdate","Gender","Family_History","Smoking","Alcohol","Dietry_Habits","Fruit","Vegetables","Fast_Food","Sweets","Sleep","Physical_Activity","Energy_Levels","Water","Juice","Soda","Height","Weight","Waist","Blood_Pressure","Glucose","Cholesterol"],
                        [birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist,blood_pressure,glucose,Cholesterol]
                        ]
    questionnaire_data = [[birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist,blood_pressure,glucose,Cholesterol]]
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
        
        with open(filename, append_write, newline='') as csvfile:
            questionnaire = csv.writer(csvfile)
            questionnaire.writerows(questionnaire_data)
    else:
        append_write = 'w' # make a new file if not
        with open(filename, append_write, newline='') as csvfile:
            questionnaire = csv.writer(csvfile)
            questionnaire.writerows(questionnaire_data_header_inc)
            
            
    return filename

def questionnaire_train():
    Title = 'Type 2 Diabeters Risk Assesment Questionaire'
    print(pyfiglet.figlet_format(Title))
    Diabetes = confirm_to_numerical(Confirm.ask("Do you have diabetes?"))
    birthdate = Prompt.ask("Please Enter your 🎂 Birthdate in Format DD-MM-YY")
    gender = gender_to_numerical(inquirer.prompt(Gender))
    #print(gender)
    family = confirm_to_numerical(Confirm.ask("Do you have any immediate family that has diabetes?"))
    smoker = confirm_to_numerical(Confirm.ask("Do you smoke 🚬 regularly?"))
    #print(smoker)
    alcohol = confirm_to_numerical(Confirm.ask("Do you drink 🍺 alcohol regularly?"))
    diet = Diet_to_numerical(inquirer.prompt(Diet))
    fruit = Prompt.ask("On a scale of 1-5 how much do you eat 🍎 fruit")
    vegtables = Prompt.ask("On a scale of 1-5 how much do you eat 🥕 vegtables")
    fast_food = Prompt.ask("On a scale of 1-5 how much do you eat out E.g. 🍔 Fast Food")
    sweets = Prompt.ask("On a scale of 1-5 how much sweet and sugary food to you eat E.g. 🍫 Chocolate, 🍩 Donuts")
    sleep = Sleep_to_numerical(inquirer.prompt(Sleep))
    activity = Prompt.ask("On a scale of 1-5 how 🏃 Phyisically active are you")
    energy_levels = Prompt.ask("On a scale of 1-5 how are your 🔋energy levels throughout the day")
    water = Prompt.ask("On a scale of 1-5 how much 💧 water do you drink throughout the day")
    juice = Prompt.ask("On a scale of 1-5 how much 🧃 juice do you drink throughout the day")
    soda = Prompt.ask("On a scale of 1-5 how much 🥤 soda do you drink throughout the day")
    height = Prompt.ask("Please Enter your 📏 Height in cm")
    weight = Prompt.ask("Please Enter your ⚖️ Weight in KG")
    waist = Prompt.ask("Please Enter your 📏 Waist Circumference in cm")
    
    filename = create_csv_train(Diabetes,birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist)
    predict.mydataset_RF(filename)
    #Blood pressure,glucose and cholestrol were removed
    #blood_pressure = Prompt.ask("19. Please input your ⚖️ blood presure result")
    #glucose = Prompt.ask("20. Please input your ⚖️ glucose result")
    #Cholesterol = Prompt.ask("21. Please input your Cholesterol result")
    return
    
    


def questionnaire_test():
    Title = 'Type 2 Diabeters Risk Assesment Questionaire'
    print(pyfiglet.figlet_format(Title))
    Diabetes = confirm_to_numerical(Confirm.ask("Do you have diabetes?"))
    birthdate = Prompt.ask("Please Enter your 🎂 Birthdate in Format DD-MM-YY")
    gender = gender_to_numerical(inquirer.prompt(Gender))
    #print(gender)
    family = confirm_to_numerical(Confirm.ask("Do you have any immediate family that has diabetes?"))
    smoker = confirm_to_numerical(Confirm.ask("Do you smoke 🚬 regularly?"))
    #print(smoker)
    alcohol = confirm_to_numerical(Confirm.ask("Do you drink 🍺 alcohol regularly?"))
    diet = Diet_to_numerical(inquirer.prompt(Diet))
    fruit = Prompt.ask("On a scale of 1-5 how much do you eat 🍎 fruit")
    vegtables = Prompt.ask("On a scale of 1-5 how much do you eat 🥕 vegtables")
    fast_food = Prompt.ask("On a scale of 1-5 how much do you eat out E.g. 🍔 Fast Food")
    sweets = Prompt.ask("On a scale of 1-5 how much sweet and sugary food to you eat E.g. 🍫 Chocolate, 🍩 Donuts")
    sleep = Sleep_to_numerical(inquirer.prompt(Sleep))
    activity = Prompt.ask("On a scale of 1-5 how 🏃 Phyisically active are you")
    energy_levels = Prompt.ask("On a scale of 1-5 how are your 🔋energy levels throughout the day")
    water = Prompt.ask("On a scale of 1-5 how much 💧 water do you drink throughout the day")
    juice = Prompt.ask("On a scale of 1-5 how much 🧃 juice do you drink throughout the day")
    soda = Prompt.ask("On a scale of 1-5 how much 🥤 soda do you drink throughout the day")
    height = Prompt.ask("Please Enter your 📏 Height in cm")
    weight = Prompt.ask("Please Enter your ⚖️ Weight in KG")
    waist = Prompt.ask("Please Enter your 📏 Waist Circumference in cm")
    
    filename = create_csv_test(Diabetes,birthdate,gender,family,smoker,alcohol,diet,fruit,vegtables,fast_food,sweets,sleep,activity,energy_levels,water,juice,soda,height,weight,waist)
    predict.mydataset_RF(filename)
    #Blood pressure,glucose and cholestrol were removed
    #blood_pressure = Prompt.ask("19. Please input your ⚖️ blood presure result")
    #glucose = Prompt.ask("20. Please input your ⚖️ glucose result")
    #Cholesterol = Prompt.ask("21. Please input your Cholesterol result")
    return



#Questionaire
@app.command("start")
def questionaire_Create():
    """
    Start the questionaire
    """
    questionnaire_test()
    
    #add recomendation 
    #add results from questionaire
    
@app.command("Input")
def questionaire_Create():
    """
    Start the questionaire
    """
    questionnaire_train()
    
    #add recomendation 
    #add results from questionaire
    
@app.command("Model")
def questionaire_Create():
    """
    Choose model type
    """
    Title = 'Choose model type'
    print(pyfiglet.figlet_format(Title))
    model = inquirer.prompt(Model)