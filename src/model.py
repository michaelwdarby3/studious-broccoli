import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class ResidualBlock(nn.Module):
    """
    Implements a residual block with two convolutional layers and a skip connection.
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Adjust the residual path to match the shape of the main path
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ProtCNN(pl.LightningModule):
    """
    Defines the ProtCNN model for sequence classification.
    """

    def __init__(self, num_classes, channels=[1, 64, 64, 128], sequence_length=128):
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
        self.accuracy = torchmetrics.Accuracy()

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        out = self.features(x)
        out = self.global_pool(out).view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        """
        Training step for the model using the given batch.
        """
        sequences, targets = batch
        out = self(sequences)
        loss = F.cross_entropy(out, targets)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(out, targets))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'train_loss',  # Optionally, specify a metric for `ReduceLROnPlateau`
            }
        }
