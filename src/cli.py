import click
from train import train_model  # Adjust this import according to your project structure
from predict import predict, load_model  # Adjust this import according to your project structure

@click.group()
def cli():
    """
    Entry point for the CLI. Groups all commands together, making them accessible through the command line.
    """
    pass

@cli.command(name="train")
@click.option('--data-dir', default='./data', type=str, help='Directory containing the data.')
@click.option('--epochs', default=10, type=int, help='Number of epochs for training.')
@click.option('--batch-size', default=32, type=int, help='Batch size for training.')
@click.option('--learning-rate', default=0.001, type=float, help='Learning rate for optimizer.')
@click.option('--accelerator', default='cpu', type=str, help='Accelerator to use: cpu, gpu, tpu, etc.')
@click.option('--devices', default=1, type=int, help='Number of devices.')
@click.option('--num-classes', default=10, type=int, help='Number of output classes.')
def train_command(data_dir, epochs, batch_size, learning_rate, accelerator, devices, num_classes):
    """
    Train the model with the specified number of epochs and data path.

    Args:
        data_dir (str): Directory containing the data.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        gpus (int): Number of GPUs.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for the model.
    """
    train_model(data_dir, epochs, batch_size, learning_rate, accelerator, devices, num_classes)

@cli.command(name="predict")
@click.option('--checkpoint-path', required=True, type=str, help='Path to the model checkpoint.')
@click.option('--sequence', required=True, type=str, help='Input sequence for making a prediction.')
@click.option('--max-len', default=128, type=int, help='Maximum length of the sequence.')
def predict_command(checkpoint_path: str, sequence: str, max_len: int):
    """
    Make a prediction based on the provided input sequence.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        sequence (str): Input sequence for making a prediction.
        max_len (int): Maximum length of the sequence.
    """
    model = load_model(checkpoint_path)
    prediction = predict(model, sequence, max_len)
    click.echo(f'Prediction: {prediction}')

if __name__ == '__main__':
    cli()
