import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load the trained model (Change filename if using Random Forest)
MODEL_PATH = "rf_model.pkl"  # or "xgb_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Define Route for Home
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bitcoin Fraud Detection API is Running!"})

# Define Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to Pandas DataFrame
        df = pd.DataFrame([data]) 

        # Ensure correct feature order (Adjust based on your dataset)
        # feature_columns = [
        #     "indegree", "outdegree", "in_btc", "out_btc", "total_btc",
        #     "mean_in_btc", "mean_out_btc"
        # ]
        # df = df[feature_columns]  # Select only relevant features

        # Convert to NumPy array for prediction
        # input_data = np.array(df)

        # Make Prediction
        prediction = model.predict(df)[0]

        # Convert prediction to JSON response
        return jsonify({"prediction": "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
