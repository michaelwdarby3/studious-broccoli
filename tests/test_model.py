import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import torch
from torch import nn
from model import ProtCNN, ResidualBlock

# Test ResidualBlock initialization
def test_residual_block_init():
    block = ResidualBlock(in_channels=1, out_channels=64)
    assert isinstance(block.conv1, nn.Conv1d)
    assert isinstance(block.bn1, nn.BatchNorm1d)
    assert isinstance(block.conv2, nn.Conv1d)
    assert isinstance(block.bn2, nn.BatchNorm1d)
    assert isinstance(block.shortcut, nn.Sequential)

# Test ResidualBlock forward pass
def test_residual_block_forward():
    block = ResidualBlock(in_channels=1, out_channels=64)
    input_tensor = torch.randn(1, 1, 128)  # Batch size 1, 1 channel, sequence length 128
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 64, 128)  # Output shape should match (batch size, out channels, sequence length)

# Test ProtCNN initialization
def test_prot_cnn_init():
    model = ProtCNN(num_classes=10)
    assert isinstance(model.features, nn.Sequential)
    assert isinstance(model.global_pool, nn.AdaptiveAvgPool1d)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.out_features == 10

# Test ProtCNN forward pass
def test_prot_cnn_forward():
    model = ProtCNN(num_classes=10)
    model.eval()  # Set model to evaluation mode
    input_tensor = torch.randn(1, 1, 128)  # Batch size 1, 1 channel, sequence length 128
    with torch.no_grad():  # Ensure no gradients are computed
        output = model.forward(input_tensor)  # Explicitly call the forward method
    assert output.shape == (1, 10)  # Output should have shape (batch size, num classes)
