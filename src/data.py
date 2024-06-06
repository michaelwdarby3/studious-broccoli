import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Callable, Optional

def read_data(partition: str, data_path: str) -> pd.DataFrame:
    """
    Read and concatenate data from a directory into a DataFrame.

    Args:
        partition (str): Specifies the data partition ('train', 'val', 'test').
        data_path (str): Path to the directory containing the data files.

    Returns:
        pd.DataFrame: DataFrame containing sequences and labels.
    """
    data = []
    partition_path = os.path.join(data_path, partition)
    for file_name in os.listdir(partition_path):
        file_path = os.path.join(partition_path, file_name)
        df = pd.read_csv(file_path)
        data.append(df)
    return pd.concat(data, ignore_index=True)

def build_vocab(data: pd.DataFrame) -> Dict[str, int]:
    """
    Build a vocabulary from the sequences in the dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the sequences.

    Returns:
        Dict[str, int]: Vocabulary mapping characters to unique indices.
    """
    vocab = set()
    for sequence in data['sequence']:
        if isinstance(sequence, str):  # Ensure that sequence is a string
            vocab.update(set(sequence))
        else:
            print(f"Non-string sequence: {sequence}, Type: {type(sequence)}")
    return {char: idx for idx, char in enumerate(vocab)}

class SequenceDataset(Dataset):
    def __init__(self, data_path: str, partition: str, max_len: int, transform: Optional[Callable] = None):
        """
        Args:
            data_path (str): Path to the root directory containing the data.
            partition (str): One of 'train', 'val', 'test'.
            max_len (int): Maximum length of the sequences after padding.
            transform (Callable, optional): Function to apply to each sample.
        """
        self.data = read_data(partition, data_path)
        self.vocab = build_vocab(self.data)
        self.max_len = max_len
        self.num_classes = len(self.data['family_id'].unique())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sequence = self.data.iloc[index]['sequence']
        sequence_encoded = torch.tensor([self.vocab[char] for char in sequence], dtype=torch.long)  # Use long dtype for sequence
        sequence_encoded = torch.nn.functional.pad(sequence_encoded, (0, self.max_len - len(sequence_encoded)))
        label = self.data.iloc[index]['label']
        return sequence_encoded.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)  # Convert sequence to float and label to long

    def default_transform(self, sequence: str, label: int, vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, int]:
        """
        Default transformation to convert sequence and label into tensors.

        Args:
            sequence (str): Input sequence.
            label (int): Label corresponding to the sequence.
            vocab (Dict[str, int]): Vocabulary mapping characters to indices.
            max_len (int): Maximum length of the sequence.

        Returns:
            Tuple[torch.Tensor, int]: Transformed sequence and label.
        """
        seq_encoded = [vocab.get(char, 0) for char in sequence]
        seq_encoded = seq_encoded[:max_len] + [0] * (max_len - len(seq_encoded))
        return torch.tensor(seq_encoded), label

class ToTensor:
    """
    Convert sequences in sample to Tensors.
    """

    def __call__(self, sequence: str, label: int, vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            sequence (str): Input sequence.
            label (int): Label corresponding to the sequence.
            vocab (Dict[str, int]): Vocabulary mapping characters to indices.
            max_len (int): Maximum length of the sequence.

        Returns:
            Tuple[torch.Tensor, int]: Transformed sequence and label.
        """
        sequence_encoded = np.array([vocab.get(char, 0) for char in sequence[:max_len]])
        sequence_encoded = np.pad(sequence_encoded, (0, max_len - len(sequence_encoded)), 'constant')
        return torch.tensor(sequence_encoded), torch.tensor(label)

if __name__ == "__main__":
    data_path = '../data/random_split'
    partition = 'train'  # Or 'val', 'test'

    train_dataset = SequenceDataset(data_path, partition, max_len=100, transform=ToTensor())

    # The dataset is now directly usable with a DataLoader for training
    # Example: train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
