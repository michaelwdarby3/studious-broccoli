import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data import SequenceDataset
from model import ProtCNN
from typing import Dict

def setup_data_loaders(data_path: str, batch_size: int, num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Set up data loaders for training, validation, and test datasets.

    Args:
        data_dir (str): Directory containing the data.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of workers for data loading.

    Returns:
        Dict[str, DataLoader]: Dictionary containing data loaders for 'train', 'val', and 'test' sets.
    """
    train_dataset = SequenceDataset(data_path, 'train', max_len=128)
    val_dataset = SequenceDataset(data_path, 'val', max_len=128)
    test_dataset = SequenceDataset(data_path, 'test', max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def train_model(data_path: str, epochs: int, batch_size: int, learning_rate: float, accelerator, devices, num_classes: int):
    """
    Train the model with the given parameters.

    Args:
        data_dir (str): Path to the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        gpus (int): Number of GPUs to use for training.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for the model.

    Raises:
        ValueError: If num_classes is not specified.
    """
    if num_classes is None:
        raise ValueError("num_classes must be specified for ProtCNN.")

    data_loaders = setup_data_loaders(data_path, batch_size)

    model = ProtCNN(num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(data_path, 'checkpoints'),
        filename='protcnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, devices=devices, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['val'])
    trainer.test(model, dataloaders=data_loaders['test'])

if __name__ == "__main__":
    DATA_DIR = "../data/random_split"
    BATCH_SIZE = 32
    MAX_EPOCHS = 25
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5
    GPUS = 1  # Set to 0 if you want to train on CPU
    NUM_CLASSES = 10  # Adjust based on your dataset

    train_model(DATA_DIR, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE, GPUS, NUM_CLASSES)
