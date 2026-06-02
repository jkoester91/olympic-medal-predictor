import pytest
import numpy as np
import pandas as pd
from src.preprocess import handle_missing_values, engineer_target_and_features

def test_handle_missing_values_imputation():
    """Test 1: Verify missing physical metrics are imputed using column medians."""
    df = pd.DataFrame({
        "Age": [20, np.nan, 30],
        "Height": [170, 180, np.nan],
        "Weight": [np.nan, 70, 80]
    })
    df_clean = handle_missing_values(df)
    assert df_clean["Age"].isnull().sum() == 0
    assert df_clean["Height"].isnull().sum() == 0
    assert df_clean["Weight"].isnull().sum() == 0
    assert df_clean["Age"].iloc[1] == 25.0

def test_handle_missing_values_immutability():
    """Test 2: Verify imputation function does not modify original dataframe."""
    df = pd.DataFrame({"Age": [20, np.nan, 30], "Height": [170, 180, 190], "Weight": [60, 70, 80]})
    df_copy = df.copy()
    _ = handle_missing_values(df)
    pd.testing.assert_frame_equal(df, df_copy)

def test_engineer_target_and_features():
    """Test 3: Verify target formulation (binary classification flag mapping)."""
    df = pd.DataFrame({
        "Age": [20, 25], "Height": [170, 180], "Weight": [60, 70],
        "Sex": ["M", "F"], "Season": ["Summer", "Winter"],
        "Medal": ["Gold", np.nan]
    })
    df_featured = engineer_target_and_features(df)
    assert "Medal_Won" in df_featured.columns
    assert df_featured["Medal_Won"].iloc[0] == 1
    assert df_featured["Medal_Won"].iloc[1] == 0

def test_keep_columns_filtering():
    """Test 4: Verify dimensional structure retains only required features."""
    df = pd.DataFrame({
        "Age": [20], "Height": [170], "Weight": [60],
        "Sex": ["M"], "Season": ["Summer"], "Medal": [np.nan],
        "NOC": ["USA"], "Games": ["2016 Summer"]
    })
    df_featured = engineer_target_and_features(df)
    expected_cols = ["Age", "Height", "Weight", "Sex", "Season", "Medal_Won"]
    assert list(df_featured.columns) == expected_cols