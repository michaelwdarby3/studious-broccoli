import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import List, Dict, Any, Tuple

class ResidualBlock(nn.Module):
    """
    Implements a residual block with two convolutional layers and a skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution. Default is 1.
        dilation (int): Dilation rate of the convolution. Default is 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ProtCNN(pl.LightningModule):
    """
    Defines the ProtCNN model for sequence classification.

    Args:
        num_classes (int): Number of output classes.
        channels (List[int]): List of channel sizes for each layer. Default is [1, 64, 64, 128].
        sequence_length (int): Length of the input sequences. Default is 128.
    """

    def __init__(self, num_classes: int, channels: List[int] = [1, 64, 64, 128], sequence_length: int = 128):
        super(ProtCNN, self).__init__()
        self.sequence_length = sequence_length

        # Define the model architecture
        layers = []
        in_channels = channels[0]
        for out_channels in channels[1:]:
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the model.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model using the given batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch containing sequences and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        sequences, targets = batch
        out = self.forward(sequences)  # Corrected to self.forward(sequences)
        loss = F.cross_entropy(out, targets)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(out, targets))
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'train_loss',  # Optionally, specify a metric for `ReduceLROnPlateau`
            }
        }
