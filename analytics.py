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
Visulise = [
    inquirer.List(
        "Visulise",
        message="Which way would you like to Visulise the dataset?",
        choices=["Heatmap"],
    ),
] 

@app.command("Visulise")
def questionaire_Create():
    """
    Visulises datasets
    """
    Title = 'Visulises datasets'
    print(pyfiglet.figlet_format(Title))
    visulise = inquirer.prompt(Visulise)