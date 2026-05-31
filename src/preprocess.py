import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_merge_data(athlete_path, noc_path):
    """Loads datasets and links athlete events to geographic regions."""
    df_athletes = pd.read_csv(athlete_path)
    df_noc = pd.read_csv(noc_path)
    # Combine datasets on National Olympic Committee codes
    df = pd.merge(df_athletes, df_noc, on="NOC", how="left")
    return df

def engineer_target_and_features(df):
    """
    Creates binary classification target and filters down 
    to essential modeling columns.
    """
    # 1 if any medal was won (Gold/Silver/Bronze), 0 if NaN
    df["Medal_Won"] = df["Medal"].notna().astype(int)
    
    # Subset to the structural columns defined in config
    keep_cols = ["Age", "Height", "Weight", "Sex", "Season", "Medal_Won"]
    return df[keep_cols].copy()

def handle_missing_values(df):
    """
    Imputes missing physical attributes using data-set medians.
    Does not modify the original input dataframe.
    """
    df_clean = df.copy()
    for col in ["Age", "Height", "Weight"]:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    return df_clean

def preprocess_pipeline(athlete_path, noc_path, test_size, random_state):
    """Executes end-to-end data preparation workflow."""
    raw_df = load_and_merge_data(athlete_path, noc_path)
    featured_df = engineer_target_and_features(raw_df)
    clean_df = handle_missing_values(featured_df)
    
    # One-hot encode categorical features cleanly
    processed_df = pd.get_dummies(clean_df, columns=["Sex", "Season"], drop_first=True)
    
    # Ensure standard Boolean columns from get_dummies convert cleanly to integers
    for col in processed_df.columns:
        if processed_df[col].dtype == bool:
            processed_df[col] = processed_df[col].astype(int)
            
    X = processed_df.drop(columns=["Medal_Won"])
    y = processed_df["Medal_Won"]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        athlete_path=config["data"]["raw_athlete_path"],
        noc_path=config["data"]["raw_noc_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    print(f"Data preprocessing complete successfully.")
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")