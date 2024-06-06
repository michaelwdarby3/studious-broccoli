import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import pandas as pd
import torch
import numpy as np
from data import read_data, build_vocab, SequenceDataset

def test_read_data():
    data_path = 'tests/mock_data'
    partition = 'train'
    df = read_data(partition, data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_build_vocab():
    data = pd.DataFrame({'sequence': ['ACGT', 'TGCA']})
    vocab = build_vocab(data)
    assert isinstance(vocab, dict)
    assert len(vocab) == 4  # A, C, G, T

def test_sequence_dataset():
    data_path = 'tests/mock_data'
    partition = 'train'
    dataset = SequenceDataset(data_path, partition, max_len=10)
    assert len(dataset) > 0
    sequence, label = dataset[0]
    print(f"Sequence: {sequence}, Label: {label}, Type of Label: {type(label)}")
    assert isinstance(sequence, torch.Tensor)
    assert isinstance(label.item(), int)  # Use .item() to get the Python int value from the tensor
