# The following code uses a logistic regression algorithm to predict if a cell is cancerous
# or benign depending on its measurements.
# The model will be able to take in numerical values of the size of a cell - membrane size,
# nucleus size ect - and will return a binary output - yes or no - depending on the data.

# Where a linear regression model aims to find the formula y = b + wx, a logistic regression
# model uses the sigmoid function to collapse values into probabilities between 0 and 1:

# p(x) = e^(b + wx) / 1 + e^(b + wx)

# The dataset being used is the Breast Cancer Wisconsin Diagnostic Dataset, containing different
# measurements for many different cells. Each row also contains information on wether the cell was
# malignent or benign.

# ---- IMPORTING LIBRARIES ---- #
import pandas as pd
# Used for loading the data into a dataframe so that it can be more easily manipulated.

import seaborn as sns
# Used for visualisng the data.

from sklearn.preprocessing import StandardScaler
# Used for normalising the data by removing the mean and scaling to a unit varience.

from sklearn.model_selection import train_test_split
# A utility to randomly split the data into a training set, used to train the model, and a testing set,
# used to evaluate the results.

from sklearn.linear_model import LogisticRegression
# A logistic regression model is a supervised ML algorithm that predicts the probability of a binary outcome
# such as 0 or 1. The sigmoid function is used to collapse data into a value between 0 and 1, and then a decision
# boundary of 0.5 splits these probabilities into distinct classes.

from sklearn.metrics import accuracy_score
# Used to calculate the percentage of correct predictions out of all predictions made by the model.
# Calculated by (True Positives + True Negatives)/Total Observations

from sklearn.metrics import classification_report
# Summarises the performance of the logistic regression model using the following metrics:
# Precision: Measures the accuracy of positive predictions
# Recall: Indicates how many actual positives were correctly identified.
# F1-Score: Balances precision and recall into a single score:
# Support: Shows the number of samples for each class.

# ---- IMPORTING DATASET ---- #
data = pd.read_csv("cells_data.csv")
# Reads the csv file "cells_data.csv" into a pandas DataFrame, where each column is a variable and each row is a value.
# This file exists in the same directory as this code, so only the name of the file is necessary to read it.


# ---- CLEANING DATASET ---- #
# As this is a real dataset, checking for NAN data is essential. For this dataset, there is a column named
# "Unnamed32" that is full of missing values.

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# Drops the "Unnamed32" column as well as the id column, as it does not contrtibute to the data of the cells.
# axis = 1:  drops a column, as opposed to axis=0 dropping a row.
# inplace = True: replaces the original dataframe with the two columns with a new dataframe without them.
# When inplace = False, a copy of the original dataframe is made instead.

data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
# diagnosis is the column of the dataset that says if a cell is malignent, denoted by "M" or benign.
# This code replaces the letter "M" with a 1 and anything else with a 0, for every row of the diagnosis
# column.


# ---- DEFINING FEATURES AND LABELS ---- #
y = data["diagnosis"]
# Creates a copy of the DataFrame containing the "diagnosis" column only as the target variable.

x = data.drop(["diagnosis"], axis=1)
# Creates a copy of the DataFrame called "x" with the diagnosis column removed.
# As x is every column except "diagnosis", the column the model will predict, x is a table of input features.


# ---- NORMALIZING THE DATASET ---- #
# This dataset contains columns that have very differnt units - some are in 10s, others in 0.1s etc.
# Normalizing the columns means all the data will affect the probability of the output equally.

scaler = StandardScaler()
# Creates a scalar object.

x_scaled = scaler.fit_transform(x)
# Fits the scaler to the feature data and transforms it.


# ---- SPLITTING DATA INTO TRAINING AND TESTING ---- #
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, random_state=0)
# Randomly splits all data points into training and testing data.
# 80% is used for training data, which is used to fit the model.
# 20% is used for testing data, which is held back in training and used to compare the models results
# with real unseen data.
# "random_state=0" fixes the random seed so the same training/testing split happens every time the code 
# is run.


# ---- MODEL TRAINING ---- #
model = LogisticRegression()
# Creates an untrained logistic regression model.

model.fit(x_train, y_train)
# Allows the model to learn the weights and bias' for the training data.


# ---- PREDICTING AND EVALUATING ---- #
y_prediction = model.predict(x_test)
# Runs the trained model on the held out test inputs to get predictions for a cell being malignent or benign

accuracy = accuracy_score(y_test, y_prediction)
print(f"Accuracy: {accuracy:.2f}")
# Calculates and prints the accuracy of the model.

print(classification_report(y_test,y_prediction))
# Prints a table of evaluation metrics that show how well the model performed.