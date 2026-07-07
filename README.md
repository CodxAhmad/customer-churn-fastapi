# Customer Churn Prediction - End-to-End MLOps Project

![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Data Versioning](https://img.shields.io/badge/Versioning-DVC-13ADC7?style=flat&logo=dvc&logoColor=white)
![Monitoring](https://img.shields.io/badge/Monitoring-Evidently_AI-FFA500?style=flat)

This project demonstrates a production-grade, end-to-end Machine Learning system for predicting telecommunications customer churn. It spans the entire MLOps lifecycle from data versioning to real-time model monitoring.

## 🚀 Live Demo

- **Frontend (Streamlit)**: [https://customer-churn-fastapi.streamlit.app](https://customer-churn-fastapi.streamlit.app)
- **Backend (FastAPI Docs)**: [https://customer-churn-fastapi-vwp1.onrender.com/docs](https://customer-churn-fastapi-vwp1.onrender.com/docs)

---

## 🏗️ Architecture

This project is built using a modern decoupled microservice architecture:

1. **Machine Learning Pipeline**: Data preprocessing, scaling, and ensemble modeling (Logistic Regression, Random Forest, XGBoost) using `scikit-learn`.
2. **Experiment Tracking**: **MLflow** integrated with **DagsHub** tracks hyperparameters, evaluation metrics (ROC-AUC, Accuracy), and serialized models.
3. **Data Versioning**: **DVC** handles versioning of large datasets and model artifacts (`scaler.pkl`, `voting_soft.pkl`), keeping Git lean.
4. **Backend (Model Serving)**: A high-performance **FastAPI** REST API serves real-time predictions. Deployed automatically to **Render**.
5. **Frontend (User Interface)**: A premium, interactive dashboard built with **Streamlit** and **Plotly**. Deployed on **Streamlit Cloud**.
6. **Model Monitoring**: **Evidently AI** simulates production data drift and generates interactive HTML reports to monitor data health and model performance degradation.

---

## 📂 Project Structure

```text
├── artifacts/             # Scalers and feature names (Tracked by DVC)
├── models/                # Trained ensemble models (Tracked by DVC)
├── reports/               # Evidently AI HTML drift and performance reports
├── src/                   # Core ML code
│   ├── data_prep.py       # Data cleaning and preprocessing
│   ├── train.py           # Model training and MLflow tracking
│   └── monitor.py         # Evidently AI drift simulation and reporting
├── main.py                # FastAPI backend service
├── streamlit_app.py       # Streamlit frontend dashboard
├── users_schema.py        # Pydantic data validation schemas
├── render.yaml            # Render deployment configuration
└── .env.example           # Environment variables template
```

---

## 💻 Local Setup

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
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 4. Pull Data and Models using DVC
```bash
dvc pull
```

### 5. Run the Backend API (FastAPI)
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 6. Run the Frontend (Streamlit)
In a new terminal window:
```bash
streamlit run streamlit_app.py
```
The UI will open at `http://localhost:8501`.

---

## 📊 Model Monitoring

This project uses **Evidently AI** to generate comprehensive HTML reports monitoring the health of the model in production. 

To simulate production drift and generate reports:
```bash
python src/monitor.py
```
This generates three reports in the `reports/` folder:
- `data_drift_report.html`: Analysis of data distribution shifts.
- `model_performance_report.html`: Classification metrics comparison.

---

## 🛠️ Tech Stack

- **Data Science**: Pandas, NumPy, Scikit-Learn, XGBoost
- **MLOps**: MLflow, DVC, DagsHub, Evidently AI
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **Deployment**: Render, Streamlit Cloud

---

## 👨‍💻 Author

**Ahmad Tanveer** 
- GitHub: [@CodxAhmad](https://github.com/CodxAhmad)