import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import torch
import numpy as np
from unittest.mock import patch
from predict import load_model, encode_sequence, prepare_input, predict

@patch('predict.ProtCNN.load_from_checkpoint')
def test_load_model(mock_load_from_checkpoint):
    checkpoint_path = 'tests/mock_checkpoint.ckpt'
    model = load_model(checkpoint_path)
    mock_load_from_checkpoint.assert_called_once_with(checkpoint_path)
    assert model.eval.called
    assert model.freeze.called

def test_encode_sequence():
    sequence = "ACGT"
    encoded_seq = encode_sequence(sequence)
    assert isinstance(encoded_seq, np.ndarray)
    assert len(encoded_seq) == 4
    assert encoded_seq.tolist() == [ord(char) for char in sequence]

def test_prepare_input():
    sequence = "ACGT"
    max_len = 10
    input_tensor = prepare_input(sequence, max_len)
    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.shape == (1, max_len)

@patch('predict.prepare_input')
def test_predict(mock_prepare_input):
    mock_prepare_input.return_value = torch.tensor([[1, 2, 3, 4]])
    model = torch.nn.Linear(10, 2)  # Simple model for testing
    sequence = "ACGT"
    max_len = 10
    with patch.object(model, 'forward', return_value=torch.tensor([[0.1, 0.9]])):
        prediction = predict(model, sequence, max_len)
        assert prediction.shape == (1, 2)
