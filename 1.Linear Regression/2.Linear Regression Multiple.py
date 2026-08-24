# The following code demonstrates a simple example of a multiple linear regression algorithm, that takes in 4
# input features and uses them to predict one output label.
# The dataset being used is the well known Combined Cycle Power Plant dataset.
# The model will predict a plant's power output from the ambient conditions of temperature, pressure, humidity
# and vacuum.
# This code follows the exact same workflow as the simple linear regression example:
# Prepare data - split data - train the model on the data - evaluate results - visualise results

# ---- IMPORTING LIBRARIES ---- #
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# ---- IMORTING DATASET ---- #
data_df = pd.read_csv("MLR_Data.csv")


# ---- DEFINING FEATURES AND LABELS ---- #
x = data_df.drop(["PE"],axis=1).values
y = data_df["PE"].values


# ---- SPLITTING DATA ---- #
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# ---- MODEL TRAINING ---- #
model = LinearRegression()
model.fit(x_train,y_train)


# ---- PREDICTING AND EVALUATING ---- #
y_predictions = model.predict(x_test)

r2_evaluation = r2_score(y_test,y_predictions)
print(f"Model R2 Score:{r2_evaluation}") 

comparison = pd.DataFrame({"Actual Value":y_test, "Predicted Value":y_predictions, "Difference":y_test-y_predictions})
print("COMPARISON OF ACTUAL VS PREDICTED VALUES")
print (comparison[0:30])

# ---- PLOTTING RESULTS ---- # 
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(),y_test.max()], color="red")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Real vs Predicted Values")
plt.show()