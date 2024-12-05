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
    
    
