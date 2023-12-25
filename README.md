# Skin Lesion Classification using Xception

This repository contains code for a skin lesion classification model using the Xception architecture. The model is trained on a dataset containing benign and malignant skin lesion images.

## Requirements
- Python 3
- TensorFlow 2
- PIL
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/skin-lesion-classification.git
   cd skin-lesion-classification
    ```
## Install Dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
- Organize your dataset into three main folders: train, test, and validation. Each of these folders should contain subfolders for each class (e.g., benign and malignant).
- Update the train_folder, test_folder, and validation_data_dir paths in the code to point to your dataset.

## Run The Code 
```bash
python skin_lesion_classification.py
```
## Results
The code will:

- Preprocess the data and visualize original and processed images.
- Train the skin lesion classification model using the Xception architecture.
- Evaluate the model's performance on the test dataset and generate accuracy and loss plots.
- Generate a confusion matrix to visualize the model's predictions.
- Make predictions on individual images.
- Evaluate the model on a directory containing images and calculate accuracy.
