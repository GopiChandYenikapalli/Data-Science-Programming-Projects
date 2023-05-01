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

RidingMowers_model = pickle.load(open('SVM_poly_model_1.pkl', "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def post():
    if request.method == "POST":
        Income = float(input("Enter the Income: "))
        Lot_Size = float(input("Enter the Lot_size: "))
        df = pd.DataFrame({'Income': [Income],'Lot_Size': [Lot_Size]})
        result = RidingMowers_model.predict(df)
        probability = RidingMowers_model.predict(df)
        return_str = f"\nThe  Income is {Income}$ and Lot_Size is {Lot_Size}, therefore it's indicated that He/She should {convert(result)} of a lawnmower\n"
        return_str += "<br><a href='/'>Back</a>"
        return return_str

    return render_template("home.html")

if __name__ == "__main__":
    app.run()

