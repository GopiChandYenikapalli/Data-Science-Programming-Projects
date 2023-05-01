import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
# import os
# exit(os.getcwd())

# load the pre-trained SVM model
RidingMowers_model = pickle.load(open('SVM_Linear_model_1.pkl', 'rb'))

print("\n*****************************************************")
print("* The RidingMowers Ownership Prediction *")
print("*****************************************************\n")

# use a while loop to repeatedly ask for inputs and generate predictions
while True:
    # ask for inputs from the user
    Income = float(input("Enter the Income: "))
    Lot_Size = float(input("Enter the Lot_size: "))

    # create a pandas dataframe from the input values
    df = pd.DataFrame({'Income': [Income], 'Lot_Size': [Lot_Size]})

    # generate a prediction for the input dataframe
    result = RidingMowers_model.predict(df)
    probability = RidingMowers_model.predict_proba(df)
    label = ("Nonowner", "owner")

    # print the input values and predicted label
    print(
        f"\n The Income is {Income}$ and Lot_Size is {Lot_Size} and prediction model indicates probability of ownership at {probability[0][1]:.4f}, therefore it's indicated that the person should be {label[result[0]]}.\n")
   
    # ask the user if they want to make another prediction
    choice = input("Do you want to make another prediction? (y/n): ")
    if choice.lower() == 'n':
        break
