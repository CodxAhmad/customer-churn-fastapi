import os
import sys
import io
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Fix Windows unicode printing error for MLflow
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_prep import load_and_prep_data

# Set up DagsHub credentials for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = "ahmadtanveer2375"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "42a1f62dcefcb1159c35638f2a9a5eab514222c7"

mlflow.set_tracking_uri("https://dagshub.com/ahmadtanveer2375/Customer_Churn_Track.mlflow/")
mlflow.set_experiment("Churn_Prediction_Ensemble")

def train_and_log():
    print("Loading data...")
    X, y = load_and_prep_data("../WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training models...")
    with mlflow.start_run():
        
        # 1. Logistic Regression
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=100, class_weight="balanced", solver="liblinear", max_iter=3000))
        ])
        
        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=10, 
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        
        # 3. XGBoost
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, 
            random_state=42, eval_metric="logloss"
        )
        
        # Soft Voting Ensemble
        soft_voting = VotingClassifier(
            estimators=[("lr", lr_pipe), ("rf", rf), ("xgb", xgb)],
            voting="soft"
        )
        
        soft_voting.fit(X_train, y_train)
        
        # Evaluate
        y_probs = soft_voting.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_probs)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        
        # Log Metrics & Params
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("ensemble_type", "soft_voting")
        mlflow.log_param("xgb_learning_rate", 0.03)
        mlflow.log_param("rf_n_estimators", 400)
        
        # Log Model
        mlflow.sklearn.log_model(
            soft_voting, 
            "voting_ensemble", 
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        
        # Save Scaler and features as artifacts locally and in MLflow
        scaler = StandardScaler()
        scaler.fit(X)
        os.makedirs("../artifacts", exist_ok=True)
        joblib.dump(scaler, "../artifacts/scaler.pkl")
        joblib.dump(X.columns.tolist(), "../artifacts/feature_names.pkl")
        
        mlflow.log_artifact("../artifacts/scaler.pkl", "preprocessing")
        mlflow.log_artifact("../artifacts/feature_names.pkl", "preprocessing")
        
        print("Training complete and logged to DagsHub!")

if __name__ == "__main__":
    train_and_log()
