import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data import SequenceDataset, reader  # Assuming these are correctly implemented
from model import ProtCNN  # Make sure this matches your model's actual class name


def setup_data_loaders(data_dir, batch_size, num_workers=0):
    """
    Existing documentation...
    """
    # Assuming the reader function is used inside the SequenceDataset to load data
    train_dataset = SequenceDataset(data_dir, 'train', max_len=128)  # max_len should match your model's input size
    val_dataset = SequenceDataset(data_dir, 'val', max_len=128)
    test_dataset = SequenceDataset(data_dir, 'test', max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    #return train_loader, val_loader, test_loader

def train_model(data_dir="../data/random_split", epochs=10, batch_size=32, learning_rate=0.001, gpus=1, num_classes=None, dropout_rate=0.5):
    """
    Train the model with the given parameters.

    Args:
        data_dir (str): Path to the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate for the model.
    """
    # Initialize dataset and DataLoader
    if num_classes is None:
        raise ValueError("num_classes must be specified for ProtCNN.")

    data_loader = setup_data_loaders(data_dir, batch_size)

    # Initialize the model with hyperparameters
    model = ProtCNN(learning_rate=learning_rate, dropout_rate=dropout_rate, num_classes=num_classes)

    # Set up PyTorch Lightning's trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(data_dir, 'checkpoints'),
        filename='protcnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, callbacks=[checkpoint_callback])
    trainer.fit(model, data_loader)
    trainer.test(model, dataloaders['test'])


if __name__ == "__main__":
    # Example default values, adjust as needed
    DATA_DIR = "../data/random_split"
    BATCH_SIZE = 32
    MAX_EPOCHS = 25
    LEARNING_RATE=0.001
    DROPOUT_RATE=0.5
    GPUS = 1  # Set to 0 if you want to train on CPU
    NUM_CLASSES = 10  # Adjust based on your dataset

    train_model(DATA_DIR, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE, GPUS, NUM_CLASSES, DROPOUT_RATE)



