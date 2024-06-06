# studious-broccoli


# Protein Classifier

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Description
Protein Classifier is a machine learning project designed to classify protein sequences using a convolutional neural network (CNN). This project leverages PyTorch and PyTorch Lightning for model training and prediction.

## Installation

### Prerequisites
- Python 3.7
- Docker (optional, for containerization)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/protein-classifier.git
   cd protein-classifier
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Build Docker image
   ```
   make build
   ```
   

##Usage
###Training the Model
To train the model, run the following command:

```sh
python src/train.py --data-dir ./data --epochs 25 --batch-size 32 --learning-rate 0.001 --gpus 1 --num-classes 10 --dropout-rate 0.5
```

###Making Predictions
To make a prediction, run the following command:

```sh
python src/predict.py --checkpoint-path path/to/your/model_checkpoint.ckpt --sequence "YOUR_SEQUENCE_HERE" --max-len 128
```

###Using the CLI
The project also includes a CLI for easier interaction:

```sh
python src/cli.py train --data-dir ./data --epochs 25 --batch-size 32 --learning-rate 0.001 --gpus 1 --num-classes 10 --dropout-rate 0.5
python src/cli.py make_prediction --checkpoint-path path/to/your/model_checkpoint.ckpt --sequence "YOUR_SEQUENCE_HERE" --max-len 128
```
###Features
- Data processing and preparation for protein sequences.
- Training a convolutional neural network (CNN) for sequence classification.
- Making predictions with the trained model.
- Command-line interface for training and prediction.
- 
##Contributing
Contributions are welcome! Please read the contributing guidelines for more information.

##License
This project is licensed under the MIT License. See the LICENSE file for details.
