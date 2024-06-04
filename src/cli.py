import click
from train import train_model  # Adjust this import according to your project structure
from predict import predict  # Adjust this import according to your project structure

@click.group()
def cli():
    """
    This function is the entry point for the CLI. It groups all commands together,
    making them accessible through the command line. This is where you can initialize
    anything that's common across commands, but often it's just used to define the command group.
    """
    pass

@cli.command()
@click.option('--data-dir', default='./data', help='Directory containing the data.')
@click.option('--epochs', default=10, type=int, help='Number of epochs for training.')
@click.option('--batch-size', default=32, help='Batch size for training.')
@click.option('--learning-rate', default=0.001, help='Learning rate for optimizer.')
@click.option('--gpus', default=1, type=int, help='Number of GPUs.')
@click.option('--num-classes', default=0.001, help='Learning rate for optimizer.')
@click.option('--dropout-rate', default=0.5, help='Dropout rate for the model.')
def train(data_dir, epochs, batch_size, learning_rate, dropout_rate):
    """
    Trains the model with the specified number of epochs and data path.
    """
    train_model(epochs=epochs, data_dir=data_dir, batch_size=batch_size, learning_rate=learning_rate, dropout_rate=dropout_rate)

@cli.command()
@click.option('--input-data', required=True, type=str, help='Input data for making a prediction.')
def predict(input_data):
    """
    Makes a prediction based on the provided input data.
    """
    prediction = predict(input_data=input_data)
    click.echo(f'Prediction: {prediction}')

if __name__ == '__main__':
    cli()
