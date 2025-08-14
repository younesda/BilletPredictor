import pandas as pd
import joblib

def load_model(MODEL_PATH):
    return joblib.load(MODEL_PATH)

def preprocess_data(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    - Impute toutes les colonnes numériques manquantes avec la médiane.
    - Garde uniquement les colonnes utilisées à l'entraînement.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    expected_cols = model.feature_names_in_
    df = df[expected_cols]

    return df
