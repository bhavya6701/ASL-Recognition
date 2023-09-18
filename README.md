# American Sign Language Recognition
This project aims to recognize American Sign Language (ASL) gestures using computer vision techniques and deep learning.
It utilizes Python, Keras, MediaPipe, and CV2 to preprocess data, train the model, test its performance, and provide a
user-friendly interface for real-time hand sign recognition.

![BAD AI 2023-09-18 1_32_25 PM](https://github.com/bhavya6701/ASL-Recognition/assets/92869151/98770781-270f-4002-a1d7-9c05f23cdb37)


## Table of Contents
- Prerequisites
- Installation
- Creating Separate Environment
- Project Structure
- Usage
- Dataset
- Model Training
- License

## Prerequisites
To use this project, you need to have the Conda installed.

Refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for instructions
on installing Conda.

## Installation
To run this project locally, follow these steps:
1. Clone the repository:
```console
git clone https://github.com/bhavya6701/hand-sign-recognition.git
```

2. Navigate to the project directory:
```console
cd hand-sign-recognition
```

3. Install the required dependencies mentioned in `requirements.txt`.
```console
conda install --file requirements.txt
```

4. Install `mediapipe` package using PyPI in Conda.
```console
pip install mediapipe
```
## Creating Separate Environment 
- If you want to create a separate environment for purpose of this project, 
1. run the below command
```console
conda env create --file env.yml
```
2. Install `mediapipe` package using PyPI in Conda.
```console
pip install mediapipe
```

## Project Structure
The project consists of the following files:

1. `data_preprocessing.py`: This file is responsible for preprocessing the data by extracting hand landmarks using
   MediaPipe. It prepares the dataset for training the model.
2. `ml_training.py`: Here, the preprocessed data is used to train the ASL recognition model using Keras. The model
   architecture, training hyperparameters, and training process can be modified in this file.
3. `ml_testing.py`: After training the model, you can evaluate its performance using this file. It loads the trained
   model and runs it on test data, providing accuracy metrics and other evaluation results. This file was used to test
   the accuracy of the generated model and is not required for the main application.
4. `main.py`: This file contains the main method where the trained ASL recognition model is utilized. It uses CV2 to
   capture video frames, extracts hand landmarks using MediaPipe, and predict output of the hand sign in real-time.

## Usage
To start the application, run the following command:
```console
python main.py
```

## Dataset
The ASL recognition model in this project was trained on the ASL Alphabet dataset available on Kaggle. The dataset
contains images of hands showing the corresponding ASL for each letter of the alphabet.

- Dataset Name: ASL Alphabet
- Dataset Source: [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Dataset Size: 87,000 images (approximately 3,000 images per letter)

There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING. SPACE, DELETE and
NOTHING - these 3 classes were not used in this project.

## Model Training
The machine learning model for digit and alphabet recognition is built using the Keras library. The model is trained on
the above mentioned dataset.

The `ml_training.py` file contains the code for model training. It loads dataset, preprocesses the data, defines the
model architecture, trains the model, and saves the trained model to a file.
To train the model, run the following command:
```console
python ml_training.py
```

Feel free to modify the model architecture, hyperparameters, or any other aspect of the project according to your
requirements. You can also extend the project by adding additional features or improving the user interface.

## License
This project is licensed under the MIT License. Feel free to modify and distribute it as per the terms of the license.
