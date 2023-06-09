import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
#import os
#exit(os.getcwd())

RidingMowers_model = pickle.load(open('SVM_Linear_model_1.pkl', "rb"))

print("\n*****************************************************")
print("* The USF Super Simple Lung Cancer Prediction Model *")
print("*****************************************************\n")
income = float(input("Enter the income: "))
Lot = float(input("Enter the income: "))

df = pd.DataFrame({'kgs_smoked': [kgs_smoked]})
result = cancer_model.predict(df)
probability = cancer_model.predict_proba(df)
treatment = ('Not Test', 'Test')
print(f"\nThe USF Simple Lung Cancer model indicates probability of cancer at {probability[0][1]:.4f}, therefore it's indicated that we should {treatment[result[0]]}.\n")
