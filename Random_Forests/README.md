# Predicting Income Class - Random Forest Classifier

## Project Description
This project aims to predict the income class (<=50K or >50K) of individuals based on their features such as age, capital gains, hours worked per week, education, and sex using the **Random Forest Classifier**.

The dataset used for this project is the **Adult Income Dataset**, which contains demographic information about adults, including their income. The goal is to create a machine learning model that can predict whether an individual's income is above or below 50K based on the available features.

## Files
- **income_prediction.py**: The main Python script that performs data preprocessing, model training, and evaluation.



## Model Details

### Step 1: Data Preprocessing
- The dataset is loaded into a pandas DataFrame.
- Columns with extra spaces are cleaned.
- The income column is binarized: 0 for <=50K and 1 for >50K.

### Step 2: Model Training and Tuning
- A Random Forest Classifier is trained on the dataset, and its performance is evaluated with different depths of trees (`max_depth`).
- The best `max_depth` is selected based on accuracy.

### Step 3: Feature Importance
- The model's feature importances are calculated and displayed to show which features are most influential in predicting income class.

## Results

- The accuracy of the default Random Forest model is printed.
- The best `max_depth` and corresponding accuracy are displayed for both original and extended features.
- A plot is generated comparing the accuracy of the model for different tree depths.

