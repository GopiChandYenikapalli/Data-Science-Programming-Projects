from flask import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
# import os
# exit(os.getcwd())

cancer_model = pickle.load(open('./data/logistic_model_example01.pkl', "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def post():
    if request.method == "POST":
        kgs_smoked = float(request.form["kgs_smoked"])
        df = pd.DataFrame({'kgs_smoked': [kgs_smoked]})
        result = cancer_model.predict(df)
        probability = cancer_model.predict_proba(df)
        treatment = ('Not Test', 'Test')
        return_str = f"\nThe USF Simple Lung Cancer model indicates probability of cancer at {probability[0][1]:.4f}, therefore it's indicated that we should {treatment[result[0]]}.\n"
        return_str += "<br><a href='/'>Back</a>"
        return return_str

    return render_template("home.html")

if __name__ == "__main__":
    app.run()

