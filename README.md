# MNIST Classification with CNN

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification, achieving >99.4% accuracy on both validation and test sets while maintaining specific architectural constraints.

## Training Progress

### Early Training (Epochs 1-4)
![Early Training Progress](./images/early_training.png)

### Final Training (Epochs 16-18)
![Final Training Progress](./images/final_training.png)

## Model Architecture

The CNN architecture is designed with the following specifications:
- Less than 20K parameters
- Uses Batch Normalization
- Implements Dropout for regularization
- Uses Global Average Pooling (no Fully Connected layers)

### Architecture Details:
----------------------------------------------------------------
Layer (type)               Output Shape         Param #
================================================================
Conv2d-1                  [-1, 16, 28, 28]     144
ReLU-2                    [-1, 16, 28, 28]     0
BatchNorm2d-3             [-1, 16, 28, 28]     32
Dropout-4                 [-1, 16, 28, 28]     0
Conv2d-5                  [-1, 20, 28, 28]     2,880
ReLU-6                    [-1, 20, 28, 28]     0
BatchNorm2d-7             [-1, 20, 28, 28]     40
Dropout-8                 [-1, 20, 28, 28]     0
MaxPool2d-9               [-1, 20, 14, 14]     0
Conv2d-10                 [-1, 24, 14, 14]     4,320
ReLU-11                   [-1, 24, 14, 14]     0
BatchNorm2d-12            [-1, 24, 14, 14]     48
Dropout-13                [-1, 24, 14, 14]     0
Conv2d-14                 [-1, 20, 14, 14]     4,320
ReLU-15                   [-1, 20, 14, 14]     0
BatchNorm2d-16            [-1, 20, 14, 14]     40
Dropout-17                [-1, 20, 14, 14]     0
MaxPool2d-18              [-1, 20, 7, 7]       0
Conv2d-19                 [-1, 24, 5, 5]       4,320
ReLU-20                   [-1, 24, 5, 5]       0
BatchNorm2d-21            [-1, 24, 5, 5]       48
Dropout-22                [-1, 24, 5, 5]       0
AvgPool2d-23              [-1, 24, 1, 1]       0
Conv2d-24                 [-1, 10, 1, 1]       250
================================================================
Total params: 16,442
Trainable params: 16,442
Non-trainable params: 0

## Key Features

1. **Data Split:**
   - Training Set: 50,000 images
   - Validation Set: 10,000 images
   - Test Set: 10,000 images

2. **Training Parameters:**
   - Epochs: 19
   - Batch Size: 32
   - Optimizer: SGD with Nesterov momentum (0.95)
   - Learning Rate: OneCycleLR (max_lr=0.12)
   - Weight Decay: 5e-5

3. **Data Augmentation:**
   - Random Rotation (±5°)
   - Random Affine (translate=0.08, scale=0.97-1.03)
   - Random Perspective (p=0.2, distortion_scale=0.1)
   - Color Jitter (brightness=0.1, contrast=0.1)

## Results

- Validation Accuracy: >99.4%
- Test Accuracy: >99.4%

## Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision
- tqdm
- torchsummary

## Model Checks

The repository includes GitHub Actions workflow that automatically verifies:
- Parameter count (< 20K)
- Use of Batch Normalization
- Implementation of Dropout
- Use of Global Average Pooling
- Absence of Fully Connected layers

## Usage

1. Clone the repository:
git clone <repository-url>
cd <repository-name>

2. Install dependencies:
pip install -r requirements.txt

3. Run the training:
python mnist_99_4.py

## License

MIT License

