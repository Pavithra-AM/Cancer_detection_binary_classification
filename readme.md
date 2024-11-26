Cancer Detection Binary Classification
This project is a machine learning-based binary classification system to detect whether a given image corresponds to a benign or malignant tumor. It uses TensorFlow and Keras for deep learning and image classification tasks.

Features
Preprocessing of images using Keras's ImageDataGenerator.
Convolutional Neural Network (CNN) architecture for classification.
Binary classification using sigmoid activation and binary_crossentropy loss.
Early stopping to optimize model training.
Training and evaluation on user-defined datasets.
Project Structure
bash
Copy code
Cancer_prediction_dataset/
│
├── cancer_prediction/         # Source code folder
│   ├── cancer_prediction.py   # Main training script
│   ├── predict.py             # Script to make predictions
│   └── requirements.txt       # List of dependencies
│
├── data/                      # Dataset folder
│   ├── train/                 # Training dataset
│   │   ├── benign/
│   │   ├── malignant/
│   │
│   ├── test/                  # Test dataset
│       ├── benign/
│       ├── malignant/
│
└── README.md                  # Project documentation
Requirements
To run this project, ensure the following dependencies are installed:

Python 3.7 or later
TensorFlow 2.x
h5py
numpy
matplotlib
Install all dependencies with:

bash
Copy code
pip install -r requirements.txt
How to Run
1. Train the Model
Run the cancer_prediction.py script to train the model:

bash
Copy code
python cancer_prediction.py
The model will be saved as cancer_detection_model.h5 after training.

2. Make Predictions
Use the predict.py script to load the trained model and make predictions:

bash
Copy code
python predict.py
Ensure the model file cancer_detection_model.h5 is in the same directory.

Dataset
The dataset is organized into train and test folders with two subfolders each: benign and malignant. Images should be placed in these folders accordingly.

Example structure:

bash
Copy code
data/
├── train/
│   ├── benign/       # Benign tumor images
│   └── malignant/    # Malignant tumor images
│
└── test/
    ├── benign/       # Benign tumor test images
    └── malignant/    # Malignant tumor test images
Model Architecture
The model uses a Convolutional Neural Network (CNN) with the following layers:

Conv2D & MaxPooling2D: Extract features and reduce spatial dimensions.
Flatten: Convert 2D feature maps to 1D.
Dense: Fully connected layers for classification.
Dropout: Prevent overfitting.
Results
After training, the model achieves high accuracy on the test dataset. Evaluation metrics:

Accuracy: Evaluated using model.evaluate().
Classification Report: Includes precision, recall, F1-score.
