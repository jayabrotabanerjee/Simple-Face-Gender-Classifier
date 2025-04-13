# GenAI & DL (PCCS681) Project for 6th Semester CSBS 
Simple vision transformer for gender classification from facial images. 
## Authors : 
| Name              | Roll No. | Enrollment No.     |
|-------------------|----------|---------------------|
| Jayabrota Banerjee| 2        | 12022002018078      |
| Aritra Ghosal     | 23       | 12022002018036      |

![prediction](pics/prediction%20of%20male.png)
![prediction](pics/prediction%20of%20female.png)
## Features
- Face detection using OpenCV's pre-trained Haar Cascade classifier
- Custom transformer architecture for gender classification
- Comprehensive data preprocessing pipeline
- Interactive GUI for real-time predictions
- Detailed performance metrics and visualization tools
- Support for both CPU and GPU acceleration

## Installation
```bash
git clone https://github.com/jayabrotabanerjee/Simple-Face-Gender-Classifier.git
cd Simple-Face-Gender-Classifier
pip install -r requirements.txt
```

### System Requirements
- Python 3.7+
- 4GB RAM minimum (8GB recommended)
- CUDA-capable GPU recommended for faster training

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
python train.py --epochs 30 --batch_size 32 --learning_rate 0.001 --save_dir models/
```

Additional training options:
- `--augment`: Enable data augmentation
- `--validation_split`: Percentage of training data to use for validation
- `--early_stopping`: Number of epochs to wait before early stopping

### 3. Evaluate the model
```bash
python evaluate.py --model_path models/gender_classifier.pth --test_dir data/test/
```
![confusion matrix](pics/confusion%20matrix.png)

![Training and Validation loss and Validation Accuracy](pics/loss%20and%20accuracy%20curve%20at%20epoch%2014.png)
### 4. Run inference on new images
```bash
python predict.py --image_path path/to/your/image.jpg --model_path models/gender_classifier.pth
```

### 5. Launch the GUI application
```bash
python gui.py --model_path models/gender_classifier.pth
```

## Model Architecture
The project implements a custom transformer architecture designed specifically for gender classification:

1. **Face Detection**: Isolates face regions using Haar Cascade classifiers
2. **Image Embedding**: Converts image patches into dense embeddings
3. **Positional Encoding**: Adds spatial information to maintain structural context
4. **Transformer Encoder**: Processes embeddings through multi-head self-attention layers
5. **Classification Head**: Makes final gender predictions using a fully connected layer

## Dependencies
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.19.5
- opencv-python>=4.5.3
- matplotlib>=3.4.2
- tqdm>=4.61.1
- Pillow>=8.2.0
- scikit-learn>=0.24.2
- tk>=0.1.0
- python-tk>=3.8.0

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
