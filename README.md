# 🚀 Telco Customer Churn Prediction: End-to-End MLOps Pipeline

![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/Versioning-DVC-13ADC7?style=flat&logo=dvc&logoColor=white)
![Evidently AI](https://img.shields.io/badge/Monitoring-Evidently_AI-FFA500?style=flat)
![Deployment](https://img.shields.io/badge/Deployment-Render_%7C_Streamlit_Cloud-blue?style=flat)

Welcome to the **End-to-End Machine Learning Operations (MLOps) Project**. This repository is not just a Jupyter Notebook; it is a fully functional, production-ready ML system built using modern software engineering and MLOps best practices. It predicts whether a telecommunications customer will churn (leave the service) based on their account and demographic data.

---

## 🔗 Live Project Links

Explore the live, deployed project and tracking dashboards:

- **🎨 Frontend UI (Streamlit):** [Interactive Churn Predictor](https://customer-churn-fastapi.streamlit.app/)
  *Try entering different customer profiles to see the gauge chart and dynamic risk alerts in action!*
- **⚙️ Backend API (FastAPI Docs):** [API Swagger Documentation](https://customer-churn-fastapi-vwp1.onrender.com/docs)
  *Test the live REST API endpoints directly from your browser.*
- **📈 Experiment Tracking (DagsHub/MLflow):** [DagsHub Repository](https://dagshub.com/ahmadtanveer2375/Customer_Churn_Track) | [MLflow UI](https://dagshub.com/ahmadtanveer2375/Customer_Churn_Track.mlflow/)
  *View the logged hyperparameters, metrics (ROC-AUC, Accuracy), and serialized model versions.*

---

## 🏗️ Project Architecture & MLOps Lifecycle

This project is decoupled into discrete services, managing the full ML lifecycle from data ingestion to post-deployment monitoring. 

### 1. Data Version Control (DVC) 🗃️
In professional environments, pushing large datasets and model files (`.csv`, `.pkl`) to Git bloats the repository. We implemented **DVC** to handle this:
- **How it works:** DVC tracks data files (like `artifacts/scaler.pkl` and `models/voting_soft.pkl`) generating tiny `.dvc` tracking files. 
- **Storage:** The heavy files are stored remotely in an S3-compatible DagsHub storage bucket, while Git only tracks the lightweight `.dvc` files. 
- **Benefit:** Anyone cloning this repository can simply run `dvc pull` to fetch the exact model state for any commit in history without overloading GitHub.

### 2. Experiment Tracking & Model Registry (MLflow + DagsHub) 🧪
Machine Learning requires dozens of iterations (tuning hyper-parameters, trying algorithms).
- **Implementation:** Integrated **MLflow** into `src/train.py`.
- **Remote Server:** Connected MLflow to a remote **DagsHub** tracking URI.
- **What is tracked:** 
  - *Parameters:* Model configurations (e.g., XGBoost `learning_rate`, Random Forest `n_estimators`).
  - *Metrics:* `roc_auc_score` and `accuracy_score`.
  - *Artifacts:* The serialized `VotingClassifier` ensemble model and `StandardScaler` are saved directly to the remote registry.

### 3. Machine Learning Engineering 🤖
- **Algorithm:** Uses a powerful **Soft Voting Ensemble Classifier** combining:
  1. `LogisticRegression` (for linear baseline interpretability)
  2. `RandomForestClassifier` (for handling non-linear interactions)
  3. `XGBoost` (for high-performance gradient boosting)
- **Class Imbalance:** Handled using `scale_pos_weight` and `class_weight="balanced"`.
- **Pipeline:** Features are mathematically scaled using `StandardScaler` ensuring robust gradient descent.

### 4. High-Performance Model Serving (FastAPI Backend) ⚡
Models in notebooks are useless if other applications can't communicate with them.
- **API Framework:** Built a RESTful API using **FastAPI** (`main.py`).
- **Validation:** Uses **Pydantic** (`users_schema.py`) to strictly validate incoming JSON payloads, ensuring the API cannot break due to bad data.
- **Deployment:** Automatically deployed to **Render** via a CI/CD pipeline triggered by Git pushes.

### 5. Interactive Dashboard (Streamlit Frontend) 📊
- **UI Architecture:** Built an aesthetic, premium dashboard using **Streamlit** (`streamlit_app.py`).
- **Communication:** The frontend makes HTTP POST requests to the FastAPI backend to fetch predictions.
- **Visuals:** Uses **Plotly** to render dynamic risk gauge charts and conditional UI alerts (Red for high risk, Green for safe).
- **Deployment:** Deployed on **Streamlit Cloud**, automatically redeploying on Git updates.

### 6. Production Model Monitoring (Evidently AI) 🕵️‍♂️
Models degrade in production as the real world changes (Data Drift). We implemented **Day-2 Operations monitoring**.
- **Implementation:** `src/monitor.py` simulates a production scenario where incoming data drifts (e.g., lower average tenure, higher monthly charges).
- **Reports:** It uses **Evidently AI** presets (`DataDriftPreset`, `ClassificationPreset`) to mathematically detect data shift.
- **Outputs:** Generates stunning interactive HTML dashboards (saved in `reports/`) analyzing which specific features drifted and how the model's accuracy degraded.

---

## 📂 Repository Structure

```text
├── artifacts/             # Scalers and feature names (Tracked by DVC)
├── models/                # Trained ensemble models (Tracked by DVC)
├── reports/               # Evidently AI HTML drift and performance reports
│   ├── data_drift_report.html
│   └── model_performance_report.html
├── src/                   # Core Machine Learning Code
│   ├── data_prep.py       # Data cleaning and categorical encoding
│   ├── train.py           # Model training and MLflow tracking
│   └── monitor.py         # Evidently AI drift simulation and reporting
├── .env.example           # Template for MLflow/DagsHub credentials
├── main.py                # FastAPI backend serving application
├── streamlit_app.py       # Streamlit frontend dashboard
├── users_schema.py        # Pydantic schemas for data validation
├── render.yaml            # Render continuous deployment configuration
└── requirements.txt       # Unified Python dependencies
```

---

## 💻 Local Development Setup

Want to run this entire MLOps pipeline on your own machine?

### 1. Clone the repository
```bash
git clone https://github.com/CodxAhmad/customer-churn-fastapi.git
cd customer-churn-fastapi
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory based on `.env.example`:
```env
# Required for running src/train.py to log to remote DagsHub
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 4. Fetch the Models (DVC)
Pull the large `.pkl` artifacts and models from remote storage:
```bash
dvc pull
```

### 5. Run the Backend API
Start the FastAPI server. It will load the models and expose the `/predict` endpoint:
```bash
uvicorn main:app --reload
```
*(The API will run at `http://127.0.0.1:8000`)*

### 6. Run the Frontend Dashboard
In a new terminal, start the UI:
```bash
streamlit run streamlit_app.py
```
*(The UI will open in your browser at `http://localhost:8501`)*

### 7. Run Model Monitoring (Evidently AI)
Generate data drift HTML reports based on simulated production drift:
```bash
python src/monitor.py
```
*(Open the newly generated `.html` files in `reports/` in your browser!)*

---

## 👨‍💻 Developed By

**Ahmad Tanveer** 
- GitHub: [@CodxAhmad](https://github.com/CodxAhmad)
- Project Focus: MLOps, Machine Learning Engineering, Full-Stack Data Science