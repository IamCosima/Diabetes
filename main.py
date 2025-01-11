import typer
from rich import print
from rich.console import Console
from rich.table import Table
import pyfiglet
from typing_extensions import Annotated
import questionaire
import train

console = Console()
app = typer.Typer(help="DiaPredic An Awsome CLI Type 2 Diabetes Risk Predictor.")
#app.add_typer(config.app,name= "Config")
#app.add_typer(train.app,name= "train")
app.add_typer(questionaire.app,name="Questionaire")

Title ='DiaPredic'
welcome_message = '[green]Welcome ' + ' :smile:'
print(pyfiglet.figlet_format(Title))
print(welcome_message)

if __name__ == "__main__":
    app()