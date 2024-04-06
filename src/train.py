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
    train_dataset = SequenceDataset(os.path.join(data_dir, 'train'))
    val_dataset = SequenceDataset(os.path.join(data_dir, 'val'))
    test_dataset = SequenceDataset(os.path.join(data_dir, 'test'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def train_model(data_dir="../data/random_split", batch_size=32, max_epochs=25, gpus=1, num_classes=None):
    """
    Encapsulates the model training process.

    Args:
        data_dir (str): Path to the directory containing the data.
        batch_size (int): Number of samples in each batch.
        max_epochs (int): Maximum number of epochs to train for.
        gpus (int): Number of GPUs to train on. 0 for CPU mode.
        num_classes (int): Number of classes in the dataset. Must be specified.
    """
    if num_classes is None:
        raise ValueError("num_classes must be specified for ProtCNN.")

    dataloaders = setup_data_loaders(data_dir, batch_size)

    model = ProtCNN(num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(data_dir, 'checkpoints'),
        filename='protcnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    trainer.test(model, dataloaders['test'])


if __name__ == "__main__":
    # Example default values, adjust as needed
    DATA_DIR = "../data/random_split"
    BATCH_SIZE = 32
    MAX_EPOCHS = 25
    GPUS = 1  # Set to 0 if you want to train on CPU
    NUM_CLASSES = 10  # Adjust based on your dataset

    train_model(DATA_DIR, BATCH_SIZE, MAX_EPOCHS, GPUS, NUM_CLASSES)
