import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Travel Analytics AI Dashboard", layout="wide")

st.title("🧳 Travel Analytics & Recommendation System")
st.caption("MLOps Multi-Model Dashboard (Flight Price, Gender Prediction & Hotel Recommender)")

tabs = st.tabs(["🏨 Hotel Recommender", "✈️ Flight Price Predictor", "👤 Gender Predictor"])

# ---------------------------------------------------------
# TAB 1: HOTEL RECOMMENDATION
# ---------------------------------------------------------
with tabs[0]:
    st.header("Hotel Recommendation Engine")
    top_n = st.slider("Select Number of Recommendations", min_value=1, max_value=10, value=5)
    
    if st.button("Get Recommendations"):
        try:
            response = requests.post(f"{API_URL}/recommend/hotel", json={"top_n": top_n})
            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                st.subheader(f"Top {top_n} Recommended Hotels")
                st.dataframe(pd.DataFrame(recs), use_container_width=True)
            else:
                st.error("API Error: Unable to fetch recommendations")
        except Exception as e:
            st.error(f"Connection Error: {e}")

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
        try:
            payload = {"distance": distance, "agency": agency}
            response = requests.post(f"{API_URL}/predict/flight", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Flight Price: ${result.get('predicted_flight_price')}")
            else:
                st.error("Model Prediction Error")
        except Exception as e:
            st.error(f"Connection Error: {e}")

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
        try:
            payload = {"age": age, "company": company}
            response = requests.post(f"{API_URL}/predict/gender", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Gender Category: {result.get('predicted_gender').upper()}")
            else:
                st.error("Model Classification Error")
        except Exception as e:
            st.error(f"Connection Error: {e}")