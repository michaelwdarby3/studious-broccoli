import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from torch.utils.data import DataLoader
from train import setup_data_loaders, train_model
from unittest.mock import patch

def test_setup_data_loaders():
    data_path = 'tests/mock_data'
    batch_size = 32
    loaders = setup_data_loaders(data_path, batch_size)
    assert isinstance(loaders['train'], DataLoader)
    assert isinstance(loaders['val'], DataLoader)
    assert isinstance(loaders['test'], DataLoader)

@patch('train.pl.Trainer.fit')
@patch('train.pl.Trainer.test')
def test_train_model(mock_fit, mock_test):
    data_path = 'tests/mock_data'
    epochs = 1
    batch_size = 32
    learning_rate = 0.001
    accelerator = 'cpu'
    devices = 1
    num_classes = 10

    train_model(data_path, epochs, batch_size, learning_rate, accelerator, devices, num_classes)
    mock_fit.assert_called_once()
    mock_test.assert_called_once()
