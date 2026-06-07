import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import json
from openai import OpenAI
from datetime import datetime

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
mlflow.set_tracking_uri("./mlruns")

api_key = os.environ.get("NEBIUS_API_KEY") 
if not api_key:
    st.error("🚨 NEBIUS_API_KEY is missing! Please make sure it is set in your .env file.")
    st.stop()

llm_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=api_key
)
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

@st.cache_resource
def load_tracked_model(model_tag_name):
    """Queries MLflow dynamically to locate a model run by its tag name."""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Olympic_Medal_Prediction")
        if experiment is None:
            return None, None
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_tag_name}'",
            order_by=["metrics.f1 DESC"]
        )
        if not runs:
            return None, None
            
        best_run = runs[0]
        best_run_id = best_run.info.run_id
        model_uri = f"runs:/{best_run_id}/model"
        return mlflow.sklearn.load_model(model_uri), best_run_id
    except Exception as e:
        return None, None

def log_inference_data(age, height, weight, sport, model_used, prediction, probability):
    """Logs prediction requests to a CSV file for production drift monitoring."""
    log_dir = "logs"
    log_file = os.path.join(log_dir, "inference_history.csv")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": int(age),
        "height_cm": int(height),
        "weight_kg": int(weight),
        "sport": sport if sport else "unknown",
        "model_architecture": model_used,
        "prediction_class": int(prediction),
        "probability": round(float(probability), 4)
    }
    
    df = pd.DataFrame([log_entry])
    df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

# ==========================================
# 2. CONVERSATIONAL MEMORY INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 3. LLM LOGIC FUNCTIONS WITH SPORT EXTRACTION
# ==========================================
def parse_input_with_history(chat_history):
    """Passes the complete chat transcript to extract metrics and the intended sport."""
    system_prompt = """
    You are an AI data extractor for an Olympic sports platform. 
    Analyze the conversation transcript provided and extract the user's Age (years), Height (cm), Weight (kg), and the target Olympic Sport.
    If they used imperial units (feet/inches or pounds) anywhere in the chat, convert them to cm and kg.
    
    CRITICAL CRITERIA:
    - If age, height, or weight are completely missing from the user's statements, you MUST set that key's value to null and set the status to "incomplete".
    - Extract the sport as a simple lowercased singular noun (e.g., "gymnastics", "swimming", "skiing", "athletics"). If not mentioned, set to null but do NOT mark the status incomplete just for the sport.
    
    CRITICAL OUTPUT FORMAT: You must respond ONLY with a valid JSON object. Do not include markdown backticks or extra prose.
    
    Match this exact schema:
    {
      "status": "complete" or "incomplete",
      "age": int or null,
      "height": int or null,
      "weight": int or null,
      "sport": string or null,
      "message": "If status is incomplete, write a friendly conversational reply asking the user for the missing metric(s). If complete, leave empty."
    }
    """
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=api_messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        parsed = json.loads(response.choices[0].message.content)
        
        if parsed.get("age") is None or parsed.get("height") is None or parsed.get("weight") is None:
            parsed["status"] = "incomplete"
            if not parsed.get("message"):
                missing = [k for k in ["age", "height", "weight"] if parsed.get(k) is None]
                parsed["message"] = f"I've noted your metrics, but I'm still missing your: {', '.join(missing)}. Could you provide that?"
                
        return parsed
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_analysis_with_llm(chat_history, parsed_data, prediction, probability, model_name):
    """Generates an engaging, contextual sports summary based on the conversation."""
    medal_status = "WOULD match historical medal winners" if prediction == 1 else "would NOT match historical medal winners"
    sport_str = f"in {parsed_data['sport']}" if parsed_data.get('sport') else ""
    
    system_prompt = f"""
    You are an expert Olympic sports analyst. 
    Review the dialogue history and the final data extracted: (Age: {parsed_data['age']}, Height: {parsed_data['height']}cm, Weight: {parsed_data['weight']}kg, Sport: {parsed_data['sport']}).
    
    Our {model_name} Machine Learning model has analyzed these physical attributes in the strict context of their chosen sport and predicted they {medal_status} with a {(probability * 100):.1f}% confidence score.
    
    Write an engaging, personal, and encouraging 2-paragraph wrap-up response explaining this result. Ensure your analysis mathematically justifies why their height-to-weight ratio matches or mismatches the demands of {parsed_data['sport']}. Direct your response directly to the user.
    """
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=api_messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating custom sports analysis: {str(e)}"

# ==========================================
# 4. STREAMLIT UI & LIVE INFERENCE
# ==========================================
st.title("🏅 Conversational AI Olympic Predictor")
st.markdown("Chat with our AI to see if you match the physical profile of an Olympic medalist!")

# SIDEBAR: Architecture Selection Hub
st.sidebar.header("⚙️ MLOps Management Hub")
selected_model_tag = st.sidebar.selectbox(
    "Select Inference Model Architecture:",
    ["deep_rf", "baseline_rf", "wide_rf", "gradient_boost", "fast_gb"]
)

model, run_id = load_tracked_model(selected_model_tag)

if model is not None:
    st.sidebar.success(f"Loaded: {selected_model_tag}\nRun ID: {run_id[:8]}")
else:
    st.sidebar.warning(f"Model '{selected_model_tag}' not active in MLflow. Using fallback.")

if st.sidebar.button("Clear Chat History 🔄"):
    st.session_state.messages = []
    st.rerun()

# Display chat transcript
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user chat submissions
if user_input := st.chat_input("Type your profile here..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🧠 Analyzing conversation history..."):
        parsed_data = parse_input_with_history(st.session_state.messages)
        
    if parsed_data.get("status") == "error":
        ai_response = f"❌ Parsing error: {parsed_data.get('message')}"
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
    elif parsed_data.get("status") == "incomplete":
        ai_response = parsed_data.get("message")
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
    elif parsed_data.get("status") == "complete":
        if model is None:
            st.error("Cannot make prediction: Selected model artifact is missing from MLflow.")
        else:
            with st.spinner(f"🤖 Computing sport-aware inference using {selected_model_tag}..."):
                h_m = parsed_data["height"] / 100.0
                bmi = parsed_data["weight"] / (h_m ** 2)
                sport = parsed_data.get("sport", "").lower()
                
                input_data = pd.DataFrame({
                    "Age": [int(parsed_data["age"])],
                    "Height": [int(parsed_data["height"])],
                    "Weight": [int(parsed_data["weight"])]
                })
                
                # Fetch baseline probability from RF
                base_prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5
                
                # --- MLOps Heuristic Interaction Layer ---
                # Penalize unviable physical frames based on engineering rules for specialized sports
                if sport in ["gymnastics", "trampoline"]:
                    if parsed_data["height"] > 185 or bmi > 26:
                        prob *= 0.15  # Heavy penalty for giant/high-mass gymnasts
                elif sport in ["skiing", "running", "marathon", "cross country"]:
                    if bmi > 28:
                        prob *= 0.25  # High mass penalty for strict high-endurance disciplines
                elif sport in ["swimming"]:
                    if parsed_data["height"] < 170:
                        prob *= 0.40  # Short stature lever penalty for elite swimming windows
                        
                # Recalculate hard threshold class post-penalty
                prediction = 1 if prob >= 0.5 else 0
                
                # Production Audit: Log everything including the parsed sport!
                log_inference_data(
                    parsed_data["age"], parsed_data["height"], parsed_data["weight"],
                    sport, selected_model_tag, prediction, prob
                )
                
                final_analysis = generate_analysis_with_llm(
                    st.session_state.messages, parsed_data, prediction, prob, selected_model_tag
                )
                
                sport_badge = sport.upper() if sport else "GENERAL OLYMPICS"
                prefix = f"📊 **Data Extracted:** Age {parsed_data['age']} | Height {parsed_data['height']}cm | Weight {parsed_data['weight']}kg\n"
                prefix += f"⚙️ **Inference Engine:** {selected_model_tag} | **Target Profile:** {sport_badge}\n\n"
                ai_response = prefix + final_analysis
                
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                if prediction == 1:
                    st.balloons()
            st.session_state.messages.append({"role": "assistant", "content": ai_response})