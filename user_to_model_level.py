import pandas as pd

def encode_input(user_data: dict, model_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([user_data])

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
    df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})

    dummies_cols = [
        "MultipleLines", "OnlineSecurity", "InternetService",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]

    df = pd.get_dummies(
        df,
        columns=dummies_cols,
        drop_first=True,
        dtype=int
    )

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    return df
