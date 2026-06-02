# 🏅 AI-Powered Olympic Medal Predictor

An end-to-end Machine Learning and Large Language Model (LLM) production platform designed to predict whether an athlete will win an Olympic medal based on their physical characteristics. The system features automated data preprocessing pipelines, experiment tracking via MLflow, containerized scaling via Docker, and an intuitive Streamlit interface powered by Nebius AI Studio.

---

## 🚀 Architecture Overview

The system acts as a bridge between raw natural language inputs and a rigid classification matrix:
1. **User Interaction (Streamlit UI):** Users input descriptions in natural language format.
2. **Feature Extraction Engine (Nebius Studio):** A Llama 3.3 70B model strictly maps text into valid JSON features. If critical metrics are absent, defensive verification routines halt inference safely.
3. **Programmatic Evaluation Layer (MLflow Client):** The client programmatically scans logs to extract the model run yielding the highest ROC-AUC.
4. **Predictive Analytics Classifier (Scikit-Learn):** The processed features feed into an optimized ensemble model, evaluating output probability weights before rendering localized responses.

---

## 🛠️ Setup & Execution Instructions

### Prerequisites
* Docker installed on host operating system.
* A valid Nebius AI Studio API Key.

### 1. Configure Local Environment Variables
Create a localized environment file mapping out credentials:
```bash
cp .env.example .env
# Open .env and add your valid token:
# NEBIUS_API_KEY="v1.CmMK..."