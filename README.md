# 🏅 AI-Powered Olympic Medal Predictor

An end-to-end Machine Learning and Large Language Model (LLM) production platform designed to predict whether an athlete's physical profile matches historical Olympic medal winners. The system features automated data preprocessing pipelines, experiment tracking via MLflow, localized containerized execution via Docker, a robust unit test suite, and an intuitive conversational Streamlit interface powered by Nebius AI Studio.

---

## 🚀 Architecture Overview

The system bridges the gap between natural language interaction and structural statistical classifiers through a modular micro-service design pattern:

1. **User Interaction Layer (Streamlit UI):** Users chat with the interface using natural language (supporting both metric and imperial measurements).
2. **Feature Extraction Engine (Nebius Studio / Llama 3.3 70B):** Converts raw conversational strings into a deterministic JSON schema. If critical physical dimensions are absent, defensive guardrails halt inference to prevent garbage predictions.
3. **MLOps Heuristic Rule-Interaction Layer:** Blends machine learning inferences with real-world physical boundaries, penalizing extreme biological anomalies (e.g., a 6'7" gymnast) before declaring an output threshold.
4. **Predictive Analytics Classifiers (Scikit-Learn):** Serves ensembled tree predictions (`Random Forest` and `Gradient Boosting`) dynamically pulled from tracked MLflow model registries.
5. **Auditing Lifecycle (Drift & Logs):** Records live transaction requests locally into an automated CSV logging structure (`logs/inference_history.csv`) for production evaluation.

---

## 💡 Reflection
Building this project was a comprehensive deep dive into the end-to-end Machine Learning lifecycle. The most challenging aspect was managing the Precision-Recall trade-off. I learned that standard accuracy metrics can be highly misleading when working with imbalanced datasets like Olympic medal history, and that threshold tuning is a powerful, low-cost way to align model behavior with business requirements.

If I had more time, I would expand the feature engineering pipeline to include longitudinal performance trends (such as previous Olympic results) to further improve the predictive power and F1-score of the model.

---

## 📈 Model Performance & Experiment Summary

Five distinct configurations were systematically logged and evaluated using an 80/20 train-test partition over historical Olympic athlete aggregates. Because our data is highly imbalanced, we prioritized **F1-Score** and **ROC-AUC** for model selection rather than simple accuracy.

| Run Name | Model Type | Hyperparameters | Key Advantage | Status / Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_rf` | Random Forest | `n_estimators=50`, `max_depth=10` | Balanced class weight base | Baselined |
| `deep_rf` | Random Forest | `n_estimators=100`, `max_depth=20` | Captures deeper splits | High variance risk |
| `wide_rf` | Random Forest | `n_estimators=200`, `max_depth=12` | General stabilizing vote | Suffers from regression to the mean |
| `gradient_boost` | Gradient Boosting | `n_estimators=50`, `lr=0.1`, `max_depth=3` | Sequential error correction | **CHAMPION MODEL** |
| `fast_gb` | Gradient Boosting | `n_estimators=100`, `lr=0.2`, `max_depth=5` | Fast learning bounds | Prone to minor overfitting |

### Selection & Threshold Optimization:
While Gradient Boosting (`gradient_boost`) was selected as the champion due to its superior handling of feature interdependence, we further optimized production performance through **custom threshold tuning**.

Given the imbalanced nature of Olympic medalists, we moved away from the default `0.50` probability threshold. Through experimental tuning, we identified **`0.55`** as the optimal threshold. This adjustment effectively balances the Precision-Recall trade-off—increasing our confidence (Precision) when predicting a "Medal Won" outcome while maintaining a robust capture rate (Recall) of true medalists.

---

## 🛠️ Installation & Execution Instructions

### Prerequisites
* Docker installed on the host operating system.
* A valid Nebius AI Studio API Key.

### 1. Configure Environment Variables
Establish a local `.env` configuration file to safely store credentials without tracking them in version control:
```bash
cp .env.example .env