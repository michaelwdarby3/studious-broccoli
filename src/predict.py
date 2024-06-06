import torch
from model import ProtCNN
import numpy as np
from typing import Union

def load_model(checkpoint_path: str) -> ProtCNN:
    """
    Load the trained model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        ProtCNN: The loaded PyTorch Lightning model.
    """
    model = ProtCNN.load_from_checkpoint(checkpoint_path)
    model.eval()  # Set the model to evaluation mode
    model.freeze()  # Freeze the model to prevent any updates to its weights
    return model

def encode_sequence(sequence: str) -> np.ndarray:
    """
    Encodes a sequence into a numerical format.

    Args:
        sequence (str): The input sequence as a string.

    Returns:
        np.ndarray: Encoded sequence.

    Note: This is a placeholder. You need to replace it with your actual sequence encoding logic.
    """
    # Placeholder implementation - replace with your actual encoding logic
    return np.array([ord(char) for char in sequence], dtype=np.int64)

def prepare_input(sequence: str, max_len: int) -> torch.Tensor:
    """
    Prepares a single sequence input for prediction.

    Args:
        sequence (str): The input sequence as a string.
        max_len (int): The fixed length to which the sequence should be padded or truncated.

    Returns:
        torch.Tensor: The processed sequence ready for model input.
    """
    encoded_seq = encode_sequence(sequence)
    padded_seq = np.zeros((max_len,))
    seq_length = min(len(encoded_seq), max_len)
    padded_seq[:seq_length] = encoded_seq[:seq_length]
    return torch.tensor(padded_seq, dtype=torch.long).unsqueeze(0)  # Add batch dimension

def predict(model: ProtCNN, sequence: str, max_len: int) -> Union[torch.Tensor, np.ndarray]:
    """
    Make a prediction for a single sequence.

    Args:
        model (ProtCNN): The trained model.
        sequence (str): The input sequence as a string.
        max_len (int): The fixed length to which the sequence should be padded or truncated.

    Returns:
        Union[torch.Tensor, np.ndarray]: The model's prediction.
    """
    input_tensor = prepare_input(sequence, max_len)
    with torch.no_grad():  # Ensure no gradients are computed to save memory
        prediction = model(input_tensor)
    return prediction

if __name__ == "__main__":
    checkpoint_path = 'path/to/your/model_checkpoint.ckpt'
    sequence = "YOUR_SEQUENCE_HERE"  # Replace with your actual sequence
    max_len = 128  # The max sequence length you’ve used in training

    model = load_model(checkpoint_path)
    prediction = predict(model, sequence, max_len)

    # Output the prediction
    print("Prediction:", prediction)
