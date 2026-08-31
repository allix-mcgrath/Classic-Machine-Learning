
# ---- IMPORTING LIBRARIES ---- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---- IMPORTING DATASET ---- #
data_df = pd.read_csv("MLR_Data.csv")


# ---- DEFINING FEATURES AND LABELS ---- #
x = data_df.drop(["PE"], axis=1).values
y = data_df["PE"].values


# ---- SPLITTING DATA ---- #
def train_test_split_manual(x, y, test_size=0.2, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = round(n * test_size)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

x_train, x_test, y_train, y_test = train_test_split_manual(x, y, test_size=0.2, seed=0)


# ---- MODEL TRAINING ---- #
def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def fit_multiple_linear_regression(X, y):
    X_b = add_intercept(X)
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

beta = fit_multiple_linear_regression(x_train, y_train)


# ---- PREDICTING AND EVALUATING ---- #
def predict(X, beta):
    return add_intercept(X) @ beta

y_predictions = predict(x_test, beta)

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

r2_evaluation = r2_score_manual(y_test, y_predictions)
print(f"Model R2 Score: {r2_evaluation}")

comparison = pd.DataFrame({
    "Actual Value": y_test,
    "Predicted Value": y_predictions,
    "Difference": y_test - y_predictions
})
print("COMPARISON OF ACTUAL VS PREDICTED VALUES")
print(comparison[0:30])

# ---- PLOTTING RESULTS ---- #
plt.figure(figsize=(15, 10))
plt.scatter(y_test, y_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Real vs Predicted Values")
plt.show()