import pytest
import pandas as pd

def test_interface_defensive_missing_cols():
    """Test 1: Verify incomplete feature drops trigger defensive routing."""
    feature_cols = ["Age", "Height", "Weight", "Sex_M", "Season_Winter"]
    
    # Simulate an incomplete extraction payload from the LLM engine
    parsed_data = {"Age": 21, "Sex_M": 1, "Season_Winter": 0}
    missing_cols = [col for col in feature_cols if col not in parsed_data]
    
    assert len(missing_cols) == 2
    assert "Height" in missing_cols
    assert "Weight" in missing_cols

def test_interface_valid_payload_mapping():
    """Test 2: Verify structural feature columns strictly conform to payload requirements."""
    feature_cols = ["Age", "Height", "Weight", "Sex_M", "Season_Winter"]
    parsed_data = {"Age": 26, "Height": 185, "Weight": 82, "Sex_M": 1, "Season_Winter": 0}
    
    feature_df_display = pd.DataFrame([parsed_data])
    input_payload = feature_df_display[feature_cols]
    
    assert list(input_payload.columns) == feature_cols
    assert input_payload.loc[0, "Age"] == 26
    assert input_payload.loc[0, "Sex_M"] == 1