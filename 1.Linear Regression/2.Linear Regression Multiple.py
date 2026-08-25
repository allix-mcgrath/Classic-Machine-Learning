# The following code demonstrates a simple example of a multiple linear regression algorithm, that takes in 4
# input features and uses them to predict one output label.
# The dataset being used is the well known Combined Cycle Power Plant dataset.
# The model will predict a plant's power output from the ambient conditions of temperature, pressure, humidity
# and vacuum.
# This code follows the exact same workflow as the simple linear regression example:
# Prepare data - split data - train the model on the data - evaluate results - visualise results


# ---- IMPORTING LIBRARIES ---- #
import pandas as pd
# Used for loading and handling data as a DataFrame - a labeled table, like a spreadsheet in code.

from sklearn.model_selection import train_test_split
# A utility to randomly split the data into a training set, used to train the model, and a testing set,
# used to evaluate the results.

from sklearn.linear_model import LinearRegression
# A linear regression model is a supervised ML algorithm that fits a straight line 
# to represent the relationship between the two variables, by minimising the prediction error.

from sklearn.metrics import r2_score
# A metric to measure how well the model fits the data when compared to a baseline model just guessing the mean value.
# A score of 1.0 means the model is making perfect predictions.
# A score of 0.0 means the model performed no better than a horizontal line at the average value.


import matplotlib.pyplot as plt
# Used for creating a scatter plot of predicted values against actual values to visualise the training results of
# the model.


# ---- IMORTING DATASET ---- #
data_df = pd.read_csv("MLR_Data.csv")
# Reads the csv file "MLR_Data.csv" into a pandas DataFrame, where each column is a variable and each row is a value.
# This file exists in the same directory as this code, so only the name of the file is necessary to read it.
# The DataFrame has 5 columns, Ambient Temperature, Vacuum, Ambient Pressure, Relative Humidity and 
# Power Output. Each column has 9569 entries.

# ---- DEFINING FEATURES AND LABELS ---- #
x = data_df.drop(["PE"],axis=1).values
# Creates a copy of the DataFrame called "x" with the PE column removed.
# axis = 1:  drops a column, as opposed to axis=0 dropping a row.
# As x is every column except "PE", the column the model will predict, x is a table of input features.

y = data_df["PE"].values
# Creates a copy of the DataFrame containing the "PE" column only as the target variable.

# ---- SPLITTING DATA ---- #
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
# Randomly splits all 9500 data points into training and testing data.
# 80% is used for training data, which is used to fit the model.
# 20% is used for testing data, which is held back in training and used to compare the models results
# with real unseen data.
# "random_state=0" fixes the random seed so the same training/testing split happens every time the code 
# is run.


# ---- MODEL TRAINING ---- #
model = LinearRegression()
# Creates an untrained linear regression model that will learn the relationship between the data in the form:
# PE = b0 + w1*feature1 + w2*feature2 + ...

model.fit(x_train,y_train)
# Allows the model to actually learn the equation of the relationship between the data.
# Uses ordinary least squares to find the best-fit coefficients and intercept that minimises the squared error between
# the predicted and actual power output.


# ---- PREDICTING AND EVALUATING ---- #
y_predictions = model.predict(x_test)
# Runs the trained model on the held out test inputs to get predicted power outputs

r2_evaluation = r2_score(y_test,y_predictions)
# Compares the predictions to the actual values and returns the coefficient of determination.
# This is a more interpretable metric than MSE as it is not in squared units of the target.

print(f"Model R2 Score:{r2_evaluation}") 

comparison = pd.DataFrame({"Actual Value":y_test, "Predicted Value":y_predictions, "Difference":y_test-y_predictions})
# Builds a new dataframe with columns for the predicted value, actual value and the difference between them.

print("COMPARISON OF ACTUAL VS PREDICTED VALUES")
print (comparison[0:30])
# Prints out the first 30 rows of the new comparison DataFrame

# ---- PLOTTING RESULTS ---- # 
plt.figure(figsize=(15,10))
# Creates a new figure with a size of 15x10 inches, in order to display all the data points.

plt.scatter(y_test,y_predictions)
# Plots the actual output values against the predicted values. If the model guessed every point correctly then every 
# point would be on the diagonal line y = x

plt.plot([y_test.min(), y_test.max()], [y_test.min(),y_test.max()], color="red")
# Draws on the line y = x in red for comparison.

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Real vs Predicted Values")
plt.show()
# Standard labeling before the graph is rendered