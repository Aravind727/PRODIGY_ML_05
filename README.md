# PRODIGY_ML_05

# Food Recognition and Calorie Estimation using Machine Learning

## Overview

This project implements a Machine Learning model capable of recognizing fruits and vegetables from images and estimating their approximate calorie content. The application combines image preprocessing, feature extraction, image classification, and calorie estimation to assist users in monitoring their dietary intake and making healthier food choices.

## Objective

Develop a machine learning model that can accurately recognize food items from images and estimate their calorie content.

## Dataset

**Fruits and Vegetables Image Recognition Dataset**

https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

The dataset contains multiple categories of fruits and vegetables organized into separate folders for each class.

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Learn
- Matplotlib

## Features

- Food image preprocessing
- Image resizing
- Feature extraction
- Multi-class food classification
- Calorie estimation
- Prediction visualization
- Accuracy evaluation
- Classification report generation

## Workflow

1. Load images from the dataset.
2. Resize images to a fixed size.
3. Convert images into feature vectors.
4. Split the dataset into training and testing sets.
5. Train a Support Vector Machine (SVM) classifier.
6. Predict food categories.
7. Estimate calorie values using a calorie lookup table.
8. Display prediction results with calorie information.

## Project Structure

```text
PRODIGY_ML_05
│
├── food_calorie_estimation.py
├── README.md
├── requirements.txt
├── food_predictions.png
└── dataset
    ├── Apple
    ├── Banana
    ├── Beetroot
    ├── ...
    └── Watermelon
```

## Files

- **food_calorie_estimation.py** – Main source code
- **README.md** – Project documentation
- **requirements.txt** – Required Python libraries
- **food_predictions.png** – Sample prediction output
- **dataset/** – Fruit and vegetable image dataset

## Output

- Displays classification accuracy
- Generates classification report
- Predicts food category
- Estimates approximate calories
- Saves prediction visualization as **food_predictions.png**

## Outcome

Successfully developed a machine learning model capable of recognizing fruits and vegetables from images and estimating their approximate calorie content.

## Internship Task

**Prodigy InfoTech Machine Learning Internship – Task 05**
