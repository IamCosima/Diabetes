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


console = Console()
app = typer.Typer()

Gender = [
    inquirer.List(
        "Gender",
        message="2. What is your gender?",
        choices=["Male", "Female","Other"],
    ),
]   
Diet = [
    inquirer.List(
        "Diet",
        message="6. Would your dietary habits be considered as 🍏 Healthy or 🍔 Unhealthy?",
        choices=["Healthy", "Unhealthy"],
    ),
] 
Sleep = [
    inquirer.List(
        "Sleep",
        message="11. How many ⏰ hours of 💤 sleep do you get a day?",
        choices=["Under 4 hours", "4-6 hours","8-10 hours","10+"],
    ),
] 


#Questionaire
@app.command("start")
def questionaire_Create():
    """
    Start the questionaire
    """
    Title = 'Type 2 Diabeters Risk Assesment Questionaire'
    print(pyfiglet.figlet_format(Title))
    birthdate = Prompt.ask("1. Please Enter your 🎂 Birthdate in Format YY-MM-DD")
    gender = inquirer.prompt(Gender)
    family = Confirm.ask("3. Do you have any immediate family that has diabetes?")
    smoker = Confirm.ask("4. Do you smoke 🚬 regularly?")
    alcohol = Confirm.ask("5. Do you drink 🍺 alcohol regularly?")
    diet = inquirer.prompt(Diet)
    fruit = Prompt.ask("7. On a scale of 1-15 how much do you eat 🍎 fruit")
    vegtables = Prompt.ask("8 .On a scale of 1-5 how much do you eat 🥕 vegtables")
    fast_food = Prompt.ask("9. On a scale of 1-5 how much do you eat out E.g. 🍔 Fast Food")
    sweets = Prompt.ask("10. On a scale of 1-5 how much sweet and sugary food to you eat E.g. 🍫 Chocolate, 🍩 Donuts")
    sleep = inquirer.prompt(Sleep)
    activity = Prompt.ask("12. On a scale of 1-5 how 🏃 Phyisically active are you")
    energy_levels = Prompt.ask("13. On a scale of 1-5 how are your 🔋energy levels throughout the day")
    water = Prompt.ask("14. On a scale of 1-5 how much 💧 water do you drink throughout the day")
    juice = Prompt.ask("15. On a scale of 1-5 how much 🧃 juice do you drink throughout the day")
    soda = Prompt.ask("16. On a scale of 1-5 how much 🥤 soda do you drink throughout the day")
    height = Prompt.ask("17. Please Enter your 📏 Height in cm")
    weight = Prompt.ask("18. Please Enter your ⚖️ Weight in KG")
    weight = Prompt.ask("19. Please Enter your 📏 Waist Circumference in cm")
    blood_pressure = Prompt.ask("19. Please input your ⚖️ blood presure result")
    glucose = Prompt.ask("20. Please input your ⚖️ glucose result")
    glucose = Prompt.ask("21. Please input your Cholesterol result")
    
    #add recomendation 
    #add results from questionaire
    
