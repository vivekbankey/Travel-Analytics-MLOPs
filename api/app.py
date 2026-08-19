from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "notebooks")

# Load Models
flight_model_path = os.path.join(MODELS_DIR, "flight_price_model.pkl")
gender_model_path = os.path.join(MODELS_DIR, "gender_classification_model.pkl")
hotel_artifacts_path = os.path.join(MODELS_DIR, "hotel_recommender_artifacts.pkl")

reg_model = joblib.load(flight_model_path) if os.path.exists(flight_model_path) else None
clf_model = joblib.load(gender_model_path) if os.path.exists(gender_model_path) else None
hotel_artifacts = joblib.load(hotel_artifacts_path) if os.path.exists(hotel_artifacts_path) else None

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Success",
        "message": "Travel Analytics API Engine is Active!"
    })

# ----------------------------------------------
# 1. Flight Price Prediction Endpoint
# ----------------------------------------------
@app.route("/predict/flight", methods=["POST"])
def predict_flight_price():
    try:
        data = request.get_json()
        if reg_model is None:
            return jsonify({"error": "Flight price model file not found!"}), 500

        if hasattr(reg_model, "feature_names_in_"):
            expected_cols = reg_model.feature_names_in_
            input_df = pd.DataFrame(0, index=[0], columns=expected_cols)
            for k, v in data.items():
                if k in input_df.columns:
                    input_df[k] = v
                elif f"{k}_{v}" in input_df.columns:
                    input_df[f"{k}_{v}"] = 1
        else:
            input_df = pd.DataFrame([data])

        prediction = reg_model.predict(input_df)[0]
        return jsonify({
            "status": "success",
            "predicted_flight_price": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------------------------
# 2. Gender Classification Endpoint (FIXED FEATURE MATCHING)
# ----------------------------------------------
@app.route("/predict/gender", methods=["POST"])
def predict_gender():
    try:
        data = request.get_json()
        age = data.get("age", 30)
        company = data.get("company", "")
        
        if clf_model is None:
            return jsonify({"error": "Gender classification model file not found!"}), 500

        # Dynamically align user inputs with model's trained One-Hot features
        if hasattr(clf_model, "feature_names_in_"):
            expected_cols = clf_model.feature_names_in_
            input_df = pd.DataFrame(0, index=[0], columns=expected_cols)
            
            if "age" in input_df.columns:
                input_df["age"] = age
                
            company_col = f"company_{company}"
            if company_col in input_df.columns:
                input_df[company_col] = 1
        else:
            input_df = pd.DataFrame([{"age": age}])

        prediction = clf_model.predict(input_df)[0]
        gender_label = "male" if prediction == 1 else "female"
        
        return jsonify({
            "status": "success",
            "predicted_gender": gender_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------------------------
# 3. Hotel Recommendation Endpoint
# ----------------------------------------------
@app.route("/recommend/hotel", methods=["POST"])
def recommend_hotel():
    try:
        data = request.get_json()
        top_n = data.get("top_n", 5)
        
        if hotel_artifacts is None:
            return jsonify({"error": "Hotel recommendation artifacts not found!"}), 500

        hotel_profiles = hotel_artifacts["hotel_profiles"]
        top_hotels = hotel_profiles.sort_values(by="booking_count", ascending=False).head(top_n)
        
        return jsonify({
            "status": "success",
            "recommendations": top_hotels[["name", "place", "average_price", "booking_count"]].to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    