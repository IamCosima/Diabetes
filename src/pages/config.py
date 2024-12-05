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


Model = [
    inquirer.List(
        "Model",
        message="Which model would you like to choose?",
        choices=["Decision trees", "Random forests"],
    ),
] 


#choose a model that was saved and display the algorithm used and the accuracy    
@app.command("Model")
def questionaire_Create():
    """
    Choose model type
    """
    Title = 'Choose model type'
    print(pyfiglet.figlet_format(Title))
    model = inquirer.prompt(Model)