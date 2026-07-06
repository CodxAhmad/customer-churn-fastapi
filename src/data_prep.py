import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prep_data(filepath="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Loads raw CSV and applies all preprocessing steps to return X and y.
    """
    df = pd.read_csv(filepath)
    
    # Clean numerical columns
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df = df.dropna()
    
    # Mappings
    gender_map = {"Male": 1, "Female": 0}
    other_map = {"Yes": 1, "No": 0}
    cols_for_maping = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    
    df["gender"] = df["gender"].map(gender_map)
    for col in cols_for_maping:
        df[col] = df[col].map(other_map).astype(int)
        
    cols_for_dummies = [
        "MultipleLines", "OnlineSecurity", "InternetService", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=cols_for_dummies, drop_first=True, dtype=int)
    
    # Drop customerID
    df = df.drop(["customerID"], axis=1)
    
    # Target mapping
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return X, y
