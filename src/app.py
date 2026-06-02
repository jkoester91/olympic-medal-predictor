import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure MLflow path to look at your local mlruns volume
mlflow.set_tracking_uri("./mlruns")

@st.cache_resource
def load_champion_model():
    """
    Queries MLflow dynamically to locate the 'deep_rf' champion run,
    extracts its exact model URI, and loads the serialized model artifact.
    """
    try:
        client = MlflowClient()
        # Find the experiment ID for Olympic Medal Prediction
        experiment = client.get_experiment_by_name("Olympic_Medal_Prediction")
        if experiment is None:
            st.error("MLflow Experiment 'Olympic_Medal_Prediction' not found.")
            return None
            
        # Query all runs in this experiment, sorted by F1-score descending
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'deep_rf'",
            order_by=["metrics.f1 DESC"]
        )
        
        if not runs:
            st.error("No training runs found for 'deep_rf' inside MLflow.")
            return None
            
        # Select the top run (highest F1-score)
        best_run = runs[0]
        best_run_id = best_run.info.run_id
        model_uri = f"runs:/{best_run_id}/model"
        
        # Load the model directly from MLflow artifacts
        model = mlflow.sklearn.load_model(model_uri)
        return model, best_run_id
        
    except Exception as e:
        st.error(f"Failed to connect to MLflow or load model: {e}")
        return None, None

# Load the model seamlessly
model, run_id = load_champion_model()

if model is not None:
    st.sidebar.success(f"Loaded Champion Model: deep_rf\n(Run ID: {run_id[:8]})")
else:
    st.sidebar.error("Running app in offline/fallback mode without MLflow model.")

# ==========================================
# STREAMLIT FRONT-END UI
# ==========================================
st.title("🏅 AI Olympic Medal Predictor")
st.markdown("Enter an athlete's physical profile below to predict their odds of taking home hardware.")

# Create a clean side-by-side visual layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Attributes")
    age = st.number_input("Age", min_value=12, max_value=75, value=24)
    height = st.number_input("Height (cm)", min_value=130, max_value=230, value=175)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=170, value=70)

with col2:
    st.subheader("Event Details (Context)")
    sex = st.selectbox("Sex", ["M", "F"])
    season = st.selectbox("Season", ["Summer", "Winter"])
    sport = st.text_input("Sport", value="Swimming")

st.markdown("---")

# The Prediction Trigger
if st.button("Predict Medal 🏆", use_container_width=True):
    if model is None:
        st.error("Cannot make prediction: Model failed to load.")
    else:
        # 1. Package the EXACT inputs your model was trained on into a DataFrame
        # (src/train.py only used Age, Height, and Weight)
        input_data = pd.DataFrame({
            "Age": [age],
            "Height": [height],
            "Weight": [weight]
        })
        
        # 2. Pass the data to your loaded MLflow model
        prediction = model.predict(input_data)
        
        # 3. Get probability score (since it's a Random Forest)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1] # Probability of class 1 (Medal)
            st.info(f"📊 **Model Confidence (Probability of Medaling): {prob:.1%}**")
        
        # 4. Display the outcome!
        if prediction[0] == 1:
            st.success(f"🥇 **Prediction:** This {int(age)}-year-old, {int(height)}cm, {int(weight)}kg athlete matches the physical profile of historical MEDAL WINNERS!")
            st.balloons()
        else:
            st.warning("📉 **Prediction:** Historical physical data suggests NO MEDAL.")