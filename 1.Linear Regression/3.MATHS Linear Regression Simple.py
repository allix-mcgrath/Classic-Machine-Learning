# This code demonstrates the same linear regression algorithm with the same data used in the file
# Linear Regression Simple.py, but uses pure maths instead of the library scikit-learn.

# For only 1 input feature, the goal of the model is to find the values of "b0" and "b1" such that
# energy_score = b0 + (b1 * hours_slept) has the minimum sum of squares error.

# For only 1 feature, calculus gives a closed form - no iteration needed - solution:
# b1 (slope)     = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
# b0 (intercept) = ȳ - b1 * x̄
# Where x̄ and ȳ are the means of x and y.

# ---- IMPORTING LIBRARIES ---- #
import random
import matplotlib.pyplot as plt


# ---- THE DATA ---- #
hours_slept = [4,5,5.5,6,6.5,7,7.5,8,8.5,9]
energy_scores = [35,42,48,52,58,65,70,79,84,90]


# ---- SPLITTING DATA ---- #
def train_test_split_manual(x,y,test_size=0.2, seed=42):
    n = len(x)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_test = round(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]

    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split_manual(hours_slept, energy_scores, test_size=0.2, seed=42)


# ---- MODEL TRAINING ---- #
def mean(values):
    return sum(values)/len(values)

def fit_linear_regression(x,y):
    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum((xi - x_mean) *(yi- y_mean) for xi, yi in zip(x,y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    b1 = numerator/denominator
    b0 = y_mean - b1 * x_mean

    return b0,b1

b0, b1 = fit_linear_regression(x_train,y_train)


# ---- PREDICTING AND EVALUATING ---- #
def predict(x, b0, b1):
    return [(b0 + b1 * xi) for xi in x]

def mean_squared_error_manual(y_true, y_pred):
    squared_error = [(yt-yp) ** 2 for yt,yp in zip(y_true,y_pred)]
    return mean(squared_error)

predictions = predict(x_test,b0,b1)
mse = mean_squared_error_manual(y_test, predictions)

print(f"Fitted line: energy = {b0:.2f} + {b1:.2f} * hours_slept")
print(f"Predicted Energy Scores: {[round(p, 2) for p in predictions]}")
print(f"Actual Energy Score: {y_test}")
print(f"Mean Squared Error: {mse:.2f}")


# ---- PLOTTING RESULTS ---- #
plt.scatter(hours_slept, energy_scores, color="steelblue", label="Real Data")
plt.plot(hours_slept, predict(hours_slept, b0, b1), color="red", label="Model Prediction")
plt.xlabel("Hours Slept")
plt.ylabel("Energy Score")
plt.title("Hours Slept vs Energy Score")
plt.legend()
plt.show()