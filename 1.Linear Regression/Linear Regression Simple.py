# The following code demonsrates a compact and simple example of a linear regression machine learning model.
# The model will aim to predict a persons 'energy score' based on the number of hours of sleep they got.
# This code also demonstrates a simple example of the standard machine learning workflow:
# Prepare data - split data - train the model on the data - evaluate results - visualise results

# ---- IMPORTING LIBRARIES---- #
import matplotlib.pyplot as plt
# Used for creating a scatter plot and regression line to visualise the training results of the model.

from sklearn.linear_model import LinearRegression
# A linear regression model is a supervised ML algorithm that fits a straight line 
# to represent the relationship between the two variables, by minimising the prediction error.

from sklearn.metrics import mean_squared_error
# Mean squared error is a metric to measure how far off the models predictions are from the actual values.

from sklearn.model_selection import train_test_split
# A utility to randomly split the data into a training set, used to train the model, and a testing set,
# used to evaluate the results.


# ---- THE DATA ---- #
hours_slept = [4,5,5.5,6,6.5,7,7.5,8,8.5,9]
energy_scores = [35,42,48,52,58,65,70,79,84,90]
# Every hours_slept[i] pairs with energy_scores[i], allowing the model to establish a relationshio between
# the two datasets.

x = [[hours] for hours in hours_slept]
# x is reshaped from a flat list to a list of single element lists - [[4], [5], [5.5]]
# This is because scikit-learn expects features as a 2D array with the shape (n_samples, n_features)
# This tells sklearn that there are many samples with only one feature.

y = energy_scores
# The target values kept as a flat list.


# ---- SPLITTING DATA ---- # 
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# Randomly splits the 10 data points into training and testing data.
# 80% is used for training data, which is used to fit the model.
# 20% is used for testing data, which is held back in training and used to compare the models results
# with real unseen data.
# "random_state=42" fixes the random seed so the same training/testing split happens every time the code 
# is run.


# ---- MODEL TRAINING ---- # 
model = LinearRegression()
# Creates an untrained model object that will learn a straight line in the form:
# energy_scores = slope * hours_slept + intercept.

model.fit(X_train,y_train)
# Allows the model to actually learn the equation of the relationship between the data.
# Uses ordinary least squares to find the slope and intercept that minimises the squared error between
# the predicted and actual energy scores on the training data.


# ---- PREDICTING AND EVALUATING ---- # 
predictions = model.predict(X_test)
# Runs the trained model on the held out test inputs to get predicted energy scores.

mse = mean_squared_error(y_test,predictions)
# Compares the predictions to the actual test values by computing the average of the squared differences.
# A lower MSE means the predictions were closer to the actual results.
# This metric is mainly used to compare the performance of different models.

print (f"Predicted Energy Scores: {predictions.round(2)}")
print (f"Actual Energy Score: {y_test}")
print (f"Mean Squared Error: {mse:.2f}")
# Outputs the predicted and real values for the 2 test points as well as the MSE to 2 decimal places.


# ---- PLOTTING ---- #
plt.scatter(x,y, color="steelblue",label="Real Data")
# Plots the original data points on a graph as blue dots.

plt.plot(x,model.predict(x), color="red",label="Model Prediction")
# Re-runs the trained model on all the x values in the dataset to produce the model's predicted line 
# across the whole range.

plt.xlabel("Hours Slept")
plt.ylabel("Energy Score")
plt.title("Hours Slept vs Energy Score")
plt.legend()
plt.show()
# This is standard labeling before the graph is rendered