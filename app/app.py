import sys
sys.path.append('src')
# https://stackoverflow.com/questions/4761041/python-import-src-modules-when-running-tests

from flask import Flask, request, jsonify

# module import

import pandas as pd
import numpy as np

from predict import encode_predict_input, predict_price

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Home Page of Application.Now we are building post API"

@app.route("/api", methods=['GET', 'POST'])
def predict():
     # Get the data from the POST request.
    json_data = request.get_json(force=True)

    # Convert json data to dataframe
    df = pd.DataFrame.from_dict(pd.json_normalize(json_data), orient='columns')
    print("-"*80)
    print(df)

    model_path = 'models/xgboost_demo.pickle'
    encoded_path = 'models/encoded_dict_demo.pickle'
    prediction = predict_price(df, encoded_path, model_path)

    # Take the first value of prediction
    output = prediction[0]
    print("price : ",output)

    return jsonify({"price" : str(output)})


if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5112',debug=True)
