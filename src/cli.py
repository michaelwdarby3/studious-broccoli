import click
from train import train_model  # Adjust this import according to your project structure
from predict import make_prediction  # Adjust this import according to your project structure

@click.group()
def cli():
    """
    This function is the entry point for the CLI. It groups all commands together,
    making them accessible through the command line. This is where you can initialize
    anything that's common across commands, but often it's just used to define the command group.
    """
    pass

@cli.command()
@click.option('--epochs', default=10, type=int, help='Number of epochs for training.')
@click.option('--data-path', default='./data', type=str, help='Path to the training data.')
def train(epochs, data_path):
    """
    Trains the model with the specified number of epochs and data path.
    """
    train_model(epochs=epochs, data_path=data_path)

@cli.command()
@click.option('--input-data', required=True, type=str, help='Input data for making a prediction.')
def predict(input_data):
    """
    Makes a prediction based on the provided input data.
    """
    prediction = make_prediction(input_data=input_data)
    click.echo(f'Prediction: {prediction}')

if __name__ == '__main__':
    cli()
