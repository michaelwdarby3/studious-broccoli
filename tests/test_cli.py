import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from click.testing import CliRunner
from cli import cli
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset


class MockModel(LightningModule):
    def __init__(self, num_classes=2):
        super(MockModel, self).__init__()
        self.layer = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss


@pytest.fixture(scope="module")
def mock_checkpoint():
    checkpoint_path = 'tests/mock_checkpoint.ckpt'
    # Create a simple mock model and save it as a checkpoint
    model = MockModel(num_classes=2)
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1)

    # Create a DataLoader with tuples for train_loader
    train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    train_loader = DataLoader(train_data, batch_size=32)

    trainer.fit(model, train_loader)
    trainer.save_checkpoint(checkpoint_path)
    yield checkpoint_path
    os.remove(checkpoint_path)


'''def test_cli_train():
    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--data-dir', 'tests/mock_data', '--epochs', 1, '--batch-size', 32,
                                 '--learning-rate', 0.001, '--accelerator', 'cpu', '--devices', 1, '--num-classes', 10])
    print(result.output)
    assert result.exit_code == 0


def test_cli_predict(mock_checkpoint):
    runner = CliRunner()
    result = runner.invoke(cli,
                           ['predict', '--checkpoint-path', mock_checkpoint, '--sequence', 'ACGT', '--max-len', 10])
    print(result.output)
    assert result.exit_code == 0'''
