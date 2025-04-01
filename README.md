# Simple Face Gender Classifier

This repository contains a basic transformer-based model for gender classification from facial images. The model is designed to be simple yet effective, using a stripped-down transformer architecture to detect faces in images and classify them by gender.

## Features

- Face detection using OpenCV's pre-trained Haar Cascade classifier
- Custom transformer architecture for gender classification
- Data preprocessing pipeline for image normalization
- Training and evaluation scripts
- Inference script for making predictions on new images

## Installation

```bash
git clone https://github.com/yourusername/simple-face-gender-classifier.git
cd simple-face-gender-classifier
pip install -r requirements.txt
```

## Usage

### 1. Prepare your dataset

Place your dataset in the `data` directory with the following structure:

```
data/
  ├── train/
  │   ├── male/
  │   │   └── *.jpg
  │   └── female/
  │       └── *.jpg
  └── test/
      ├── male/
      │   └── *.jpg
      └── female/
          └── *.jpg
```

### 2. Train the model

```bash
python train.py --epochs 30 --batch_size 32
```

### 3. Evaluate the model

```bash
python evaluate.py --model_path models/gender_classifier.pth
```

### 4. Run inference on new images

```bash
python predict.py --image_path path/to/your/image.jpg --model_path models/gender_classifier.pth
```

## Project Structure

```
.
├── data/                      # Dataset directory
├── models/                    # Saved models directory
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset and data loader utilities
│   ├── model.py               # Transformer model architecture
│   ├── face_detector.py       # Face detection utilities
│   └── utils.py               # Helper functions
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── predict.py                 # Inference script
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Model Architecture

This project implements a simplified transformer architecture for gender classification:

1. **Image Embedding**: Converts image patches into embeddings
2. **Positional Encoding**: Adds positional information to the embeddings
3. **Transformer Encoder**: Processes the embeddings with self-attention
4. **Classification Head**: Predicts gender from the encoded features

## Requirements

- Python 3.7+
- PyTorch 1.9+
- OpenCV 4.5+
- NumPy
- Matplotlib
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.
