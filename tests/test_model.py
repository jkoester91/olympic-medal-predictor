import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def test_model_prediction_shape_and_type():
    """Test 1: Verify trained model produces valid binary evaluation formats."""
    X_train = pd.DataFrame({
        "Age": [22, 25, 30], "Height": [180, 175, 165], "Weight": [75, 70, 60],
        "Sex_M": [1, 0, 1], "Season_Winter": [0, 1, 0]
    })
    y_train = pd.Series([1, 0, 0])
    
    model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    X_test = pd.DataFrame({
        "Age": [26], "Height": [185], "Weight": [82],
        "Sex_M": [1], "Season_Winter": [0]
    })
    
    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)
    assert preds[0] in [0, 1]

def test_model_performance_threshold():
    """Test 2: Verify predictions meet baseline class logic limits."""
    X_train = pd.DataFrame({
        "Age": [25]*10 + [30]*10,
        "Height": [190]*10 + [160]*10,
        "Weight": [90]*10 + [50]*10,
        "Sex_M": [1]*10 + [0]*10,
        "Season_Winter": [0]*20
    })
    y_train = pd.Series([1]*10 + [0]*10)
    
    model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    test_sample = pd.DataFrame({
        "Age": [25], "Height": [192], "Weight": [93],
        "Sex_M": [1], "Season_Winter": [0]
    })
    pred = model.predict(test_sample)
    assert pred[0] == 1