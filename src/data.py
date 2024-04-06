import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def reader(partition, data_path):
    """
    Reads and processes files from a directory into a DataFrame.

    Args:
        partition (str): Specifies the data partition ('train', 'val', 'test').
        data_path (str): Path to the directory containing the data files.

    Returns:
        DataFrame: Contains sequences and labels.
    """
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        file_path = os.path.join(data_path, partition, file_name)
        df = pd.read_csv(file_path)
        data.append(df)
    all_data = pd.concat(data, ignore_index=True)
    return all_data


class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for loading sequence data directly from files.
    """

    def __init__(self, word2id, fam2label, max_len, data_path, split, sequence_col='sequence', label_col='label', transform=None):
        """
        Args:
            partition (str): Specifies the data partition ('train', 'val', 'test').
            data_path (str): Path to the directory containing the data files.
            sequence_col (str): Column name for sequences.
            label_col (str): Column name for labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data, self.label = reader(split, data_path)
        self.sequence_col = sequence_col
        self.label_col = label_col
        self.transform = transform

        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])
        sample = {'sequence': sequence, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def preprocess(self, text):
        seq = []

        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id), )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq


class ToTensor:
    """
    Convert sequences in sample to Tensors.
    """

    def __call__(self, sample):
        sequence, label = sample['sequence'], sample['label']
        # Place your sequence encoding logic here
        sequence_encoded = np.array([ord(c) for c in sequence])  # Example placeholder
        return {'sequence': torch.from_numpy(sequence_encoded),
                'label': torch.tensor(label)}


# Example usage is now simplified
if __name__ == "__main__":
    data_path = '../data/random_split'
    partition = 'train'  # Or 'val', 'test'

    train_dataset = SequenceDataset(partition, data_path, transform=ToTensor())

    # The dataset is now directly usable with a DataLoader for training
    # Example: train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
