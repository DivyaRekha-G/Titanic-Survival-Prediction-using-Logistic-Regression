# Titanic-Survival-Prediction-using-Logistic-Regression
This project uses Logistic Regression to predict passenger survival on the Titanic based on various features such as age, fare, gender, and class. The Titanic dataset is a classic dataset used in machine learning for classification tasks
Dataset

## The dataset used in this project comes from the Kaggle Titanic competition. 

It contains the following key features:

PassengerId: Unique ID for each passenger

Survived: Target variable (0 = No, 1 = Yes)

Pclass: Ticket class (1st, 2nd, 3rd)

Name: Passenger name

Sex: Gender

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Ticket: Ticket number

Fare: Passenger fare

Cabin: Cabin number (many missing values)

Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# Machine Learning Approach

## Steps Implemented:

Data Preprocessing

Handling missing values (imputing missing ages and embarked ports)

Converting categorical variables (sex and embarked) into numerical values

Feature scaling using StandardScaler

Splitting data into training and testing sets

## Exploratory Data Analysis (EDA)

Survival Distribution: Analyzing the overall survival rate of passengers.

Survival Rate by Gender: Comparing male and female survival rates.

Survival Rate by Passenger Class: Identifying how ticket class impacted survival.

Age Distribution: Visualizing the age distribution of passengers.

Correlation Analysis: Checking feature correlations with survival using a heatmap.

## Model Training & Evaluation

Implementing Logistic Regression using sklearn

Evaluating model performance with accuracy, precision, recall, and confusion matrix

ROC curve analysis

Dependencies

## The following Python libraries are required:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

## Results

The model achieved an accuracy of approximately 80.44% on the test dataset.

Gender and class were found to be the most significant predictors of survival.

The confusion matrix and classification report provided insights into precision and recall scores.

## Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook or Python script to train and test the model.

## Conclusion

Logistic Regression provides a good baseline for classifying survival on the Titanic dataset. Further improvements can be made using feature engineering, hyperparameter tuning, and ensemble methods.



Author

Your Name - DivyaRekha G
