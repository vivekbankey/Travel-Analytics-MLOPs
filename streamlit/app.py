import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Travel Analytics AI Dashboard", layout="wide")

st.title("🧳 Travel Analytics & Recommendation System")
st.caption("MLOps Multi-Model Dashboard (Flight Price, Gender Prediction & Hotel Recommender)")

# Model and Artifact Paths Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")

flight_model_path = os.path.join(NOTEBOOKS_DIR, "flight_price_model.pkl")
gender_model_path = os.path.join(NOTEBOOKS_DIR, "gender_classification_model.pkl")
hotel_artifacts_path = os.path.join(NOTEBOOKS_DIR, "hotel_recommender_artifacts.pkl")

# Load trained models directly
@st.cache_resource
def load_models():
    reg_model = joblib.load(flight_model_path) if os.path.exists(flight_model_path) else None
    clf_model = joblib.load(gender_model_path) if os.path.exists(gender_model_path) else None
    hotel_artifacts = joblib.load(hotel_artifacts_path) if os.path.exists(hotel_artifacts_path) else None
    return reg_model, clf_model, hotel_artifacts

reg_model, clf_model, hotel_artifacts = load_models()

tabs = st.tabs(["🏨 Hotel Recommender", "✈️ Flight Price Predictor", "👤 Gender Predictor"])

# ---------------------------------------------------------
# TAB 1: HOTEL RECOMMENDATION
# ---------------------------------------------------------
with tabs[0]:
    st.header("Hotel Recommendation Engine")
    top_n = st.slider("Select Number of Recommendations", min_value=1, max_value=10, value=5)
    
    if st.button("Get Recommendations"):
        if hotel_artifacts is not None:
            hotel_profiles = hotel_artifacts.get("hotel_profiles")
            top_hotels = hotel_profiles.sort_values(by="booking_count", ascending=False).head(top_n)
            st.subheader(f"Top {top_n} Recommended Hotels")
            st.dataframe(top_hotels[["name", "place", "average_price", "booking_count"]], use_container_width=True)
        else:
            st.error("Hotel recommendation model file not found in repository!")

# ---------------------------------------------------------
# TAB 2: FLIGHT PRICE PREDICTION
# ---------------------------------------------------------
with tabs[1]:
    st.header("Flight Price Regression Model")
    st.info("Input journey details to predict estimated flight price.")
    
    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("Flight Distance (km)", min_value=50, max_value=10000, value=500)
    with col2:
        agency = st.selectbox("Travel Agency", ["Agency A", "Agency B", "Agency C"])

    if st.button("Predict Flight Price"):
        if reg_model is not None:
            if hasattr(reg_model, "feature_names_in_"):
                expected_cols = reg_model.feature_names_in_
                input_df = pd.DataFrame(0, index=[0], columns=expected_cols)
                if "distance" in input_df.columns:
                    input_df["distance"] = distance
            else:
                input_df = pd.DataFrame([{"distance": distance}])
                
            pred = reg_model.predict(input_df)[0]
            st.success(f"Predicted Flight Price: ${round(float(pred), 2)}")
        else:
            st.error("Flight price prediction model file not found!")

# ---------------------------------------------------------
# TAB 3: GENDER CLASSIFICATION
# ---------------------------------------------------------
with tabs[2]:
    st.header("Gender Classification Model")
    st.info("Classify demographic profile based on age and company structure.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("User Age", min_value=18, max_value=80, value=30)
    with col2:
        company = st.text_input("Company Name", value="Acme Corp")
        
    if st.button("Predict Demographics"):
        if clf_model is not None:
            if hasattr(clf_model, "feature_names_in_"):
                expected_cols = clf_model.feature_names_in_
                input_df = pd.DataFrame(0, index=[0], columns=expected_cols)
                if "age" in input_df.columns:
                    input_df["age"] = age
                if f"company_{company}" in input_df.columns:
                    input_df[f"company_{company}"] = 1
            else:
                input_df = pd.DataFrame([{"age": age}])

            pred = clf_model.predict(input_df)[0]
            label = "MALE" if pred == 1 else "FEMALE"
            st.success(f"Predicted Gender Category: {label}")
        else:
            st.error("Gender classification model file not found!")