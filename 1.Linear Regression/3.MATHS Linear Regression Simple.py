# This code demonstrates the same linear regression algorithm with the same data used in the file
# Linear Regression Simple.py, but uses pure maths instead of the library scikit-learn.

# For only 1 input feature, the goal of the model is to find the values of "b0" and "b1" such that
# energy_score = b0 + (b1 * hours_slept) has the minimum sum of squares error.

# For only 1 feature, calculus gives a closed form - no iteration needed - solution:
# b1 (slope)     = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
# b0 (intercept) = ȳ - b1 * x̄
# Where x̄ and ȳ are the means of x and y.
# The slope measures how much x and y move together - covarience.
# The intercept shows where the equation passes through 0 on the x asis.


# ---- IMPORTING LIBRARIES ---- #
import random
# Used for shuffling data before the train/test split.

import matplotlib.pyplot as plt
# Used for creating a scatter plot and regression line to visualise the training results of the model.


# ---- THE DATA ---- #
hours_slept = [4,5,5.5,6,6.5,7,7.5,8,8.5,9]
energy_scores = [35,42,48,52,58,65,70,79,84,90]
# Every hours_slept[i] pairs with energy_scores[i], allowing the model to establish a relationshio between
# the two datasets.


# ---- SPLITTING DATA ---- #
def train_test_split_manual(x,y,test_size=0.2, seed=42):
# This function takes in the two datasets, hours_slept and energy_score and randomly splits them into training
# and testing data.
# 80% is used for training data, which is used to fit the model.
# 20% is used for testing data, which is held back in training and used to compare the models results
# with real unseen data.

    n = len(x)
    # The length of the dataset x (and y), in this case 9

    indices = list(range(n))
    # Creates a list of numbers from 0-9, with each index corresponding to a datapoint.

    random.seed(seed)
    # With the seed set to a cetain value, the same testing/training split will occur every time the code is run.

    random.shuffle(indices)
    # Reorders the list indices, so the list now reads:
    # [7,3,2,8,5,6,9,4,0,1]

    n_test = round(n * test_size)
    # n_test = round(10 * 0.2) = 2
    # Shows that the first two shuffled indices will become the test set, the remaining 8 will become
    # the training set.

    test_idx = indices[:n_test]
    # The list test_idx becomes [7,3], corresponding with the first two shuffled values of indices.

    train_idx = indices[n_test:]
    # train_idx becomes [2,8,5,6,9,4,0,1]

    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    # These list comprehensions look up the actual values that correspond to the indices.
    # This results in each list containing the correct number of randomly ordered data points from the correct
    # corresponding dataset.

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split_manual(hours_slept, energy_scores, test_size=0.2, seed=42)
# Runs the above function on the two datasets, creating the training and testing datasets.


# ---- MODEL TRAINING ---- #
def mean(values):
    return sum(values)/len(values)
    # Returns the arithmetic mean of a list of numbers.

def fit_linear_regression(x,y):
# This function implements the closed-form least squares formula for a line:
# y = b0 + b1*x
# It takes in two datasets, x and y, which correspond to hours_slept and energy_scores respectively.

    x_mean = mean(x)
    y_mean = mean(y)
    # Calculates the mean of the two datasets x and y.

    numerator = sum((xi - x_mean) *(yi- y_mean) for xi, yi in zip(x,y))
    # The covarience-like sum:how far each x and y point are from their respective means, multiplied together
    # amd then summed.
    # This represents how much x and y vary together.

    denominator = sum((xi - x_mean) ** 2 for xi in x)
    # The varience-like sum of x: how much x varies on its own.

    b1 = numerator/denominator
    # Works out the slope/

    b0 = y_mean - b1 * x_mean
    # Works out the intercept. This is derived from the fact that the regression line always passes through the point
    # (x_mean, y_mean)

    return b0,b1

b0, b1 = fit_linear_regression(x_train,y_train)


# ---- PREDICTING AND EVALUATING ---- #
def predict(x, b0, b1):
    return [(b0 + b1 * xi) for xi in x]
    # Applies the fitted model to any list with x values.

def mean_squared_error_manual(y_true, y_pred):
# This function finds the MSE between the actual and predicted output values.
    
    squared_error = [(yt-yp) ** 2 for yt,yp in zip(y_true,y_pred)]
    return mean(squared_error)

predictions = predict(x_test,b0,b1)
# Runs the model on the previously held out test data, to produce a list of predicted outputs.

mse = mean_squared_error_manual(y_test, predictions)
# Produces the MSE for the held out test results and the predicted data.

print(f"Fitted line: energy = {b0:.2f} + {b1:.2f} * hours_slept")
print(f"Predicted Energy Scores: {[round(p, 2) for p in predictions]}")
print(f"Actual Energy Score: {y_test}")
print(f"Mean Squared Error: {mse:.2f}")
# Outputs the predicted and real values for the 2 test points as well as the MSE to 2 decimal places.

# ---- PLOTTING RESULTS ---- #
plt.scatter(hours_slept, energy_scores, color="steelblue", label="Real Data")
# Plots the original data points on a graph as blue dots.

plt.plot(hours_slept, predict(hours_slept, b0, b1), color="red", label="Model Prediction")
# Re-runs the trained model on all the x values in the dataset to produce the model's predicted line 
# across the whole range.

plt.xlabel("Hours Slept")
plt.ylabel("Energy Score")
plt.title("Hours Slept vs Energy Score")
plt.legend()
plt.show()
# This is standard labeling before the graph is rendered