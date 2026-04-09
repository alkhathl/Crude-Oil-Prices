import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from process import (
    combine_price_and_production,
    merge_events,
    build_summary_table,
    build_detailed_production,
    build_detailed_events
)

def make_oil_prices():
    return pd.DataFrame({
        "date": ["2022-01", "2022-02", "2022-03"],
        "oil_price": [77.5, 97.2, 117.3]
    })

def make_oil_prod():
    return pd.DataFrame({
        "date":           ["2022-01", "2022-01", "2022-02", "2022-02"],
        "country":        ["Saudi Arabia", "Russia", "Saudi Arabia", "Russia"],
        "production_mbd": [10.1, 10.5, 10.2, 9.9],
        "opec_member":    ["Yes", "No", "Yes", "No"]
    })

def make_acled():
    return pd.DataFrame({
        "date":           ["2022-01", "2022-01", "2022-01", "2022-02"],
        "country":        ["Iraq", "Iraq", "Saudi Arabia", "Russia"],
        "event_type":     ["Battles", "Battles", "Explosions/Remote violence", "Protests"],
        "sub_event_type": ["Armed clash", "Armed clash", "Air/drone strike", "Peaceful protest"],
        "disorder_type":  ["Political violence", "Political violence", "Political violence", "Political violence"]
    })

# Test API
def test_get_oil_prices_returns_dataframe():
    from load import get_oil_prices
    df = get_oil_prices()
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert "date" in df.columns, "Missing 'date' column"
    assert "oil_price" in df.columns, "Missing 'oil_price' column"

def test_get_oil_production_returns_dataframe():
    from load import get_oil_production
    df = get_oil_production()
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    for col in ["date", "country", "production_mbd", "opec_member"]:
        assert col in df.columns, f"Missing column: {col}"

# Test all countries is captured
def test_get_oil_production_has_expected_countries():
    from load import get_oil_production
    from config import COUNTRIES
    df = get_oil_production()
    expected = set(COUNTRIES.values())
    actual = set(df["country"].unique())
    missing = expected - actual
    assert not missing, f"Missing countries in production data: {missing}"

# Test oil price data quality
def test_oil_price_values_are_numeric():
    from load import get_oil_prices
    df = get_oil_prices()
    df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")
    assert df["oil_price"].notna().all(), "oil_price contains nulls"
    assert (df["oil_price"] > 0).all(), "oil_price should be positive"


# Test oil production data quality
def test_production_values_are_numeric():
    from load import get_oil_production
    df = get_oil_production()
    assert df["production_mbd"].notna().all(), "production_mbd contains nulls"
    assert (df["production_mbd"] > 0).all(), "production_mbd should be positive"

# Test Merged datasets and sample countries/events
def test_combine_price_and_production_columns():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    for col in ["date", "country", "production_mbd", "oil_price"]:
        assert col in merged.columns, f"Missing column: {col}"

def test_combine_price_and_production_row_count():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    assert len(merged) == len(make_oil_prod()), "Row count should match production rows"

def test_combine_price_and_production_types():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    assert pd.api.types.is_float_dtype(merged["oil_price"]), "oil_price should be float"
    assert pd.api.types.is_float_dtype(merged["production_mbd"]), "production_mbd should be float"

def test_merge_events_adds_event_count():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    result = merge_events(merged, make_acled())
    assert "event_count" in result.columns, "Missing event_count column"

def test_merge_events_no_nulls_in_event_count():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    result = merge_events(merged, make_acled())
    assert result["event_count"].notna().all(), "event_count should not have nulls"

def test_merge_events_correct_count():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    result = merge_events(merged, make_acled())
    row = result[(result["date"] == "2022-01") & (result["country"] == "Saudi Arabia")]
    assert row["event_count"].values[0] == 1.0

def test_build_summary_table_has_summary_id():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    merged = merge_events(merged, make_acled())
    summary = build_summary_table(merged)
    assert "summary_id" in summary.columns

def test_build_summary_table_one_row_per_date():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    merged = merge_events(merged, make_acled())
    summary = build_summary_table(merged)
    assert summary["date"].nunique() == len(summary), "Duplicate dates in summary"

# Testing Primary Keys
def test_build_detailed_production_has_production_id():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    merged = merge_events(merged, make_acled())
    summary = build_summary_table(merged)
    detailed = build_detailed_production(merged, summary)
    assert "production_id" in detailed.columns

def test_build_detailed_events_has_event_id():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    merged = merge_events(merged, make_acled())
    summary = build_summary_table(merged)
    detailed_prod = build_detailed_production(merged, summary)
    detailed_events = build_detailed_events(make_acled(), summary, detailed_prod)
    assert "event_id" in detailed_events.columns

# Testing all events are captured
def test_build_detailed_events_row_count():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    merged = merge_events(merged, make_acled())
    summary = build_summary_table(merged)
    detailed_prod = build_detailed_production(merged, summary)
    acled = make_acled()
    detailed_events = build_detailed_events(acled, summary, detailed_prod)
    assert len(detailed_events) == len(acled), "Row count should match raw ACLED"


# Testing how null values are handled

def test_combine_handles_missing_price_dates():
    prices = pd.DataFrame({"date": ["2022-01"], "oil_price": [77.5]})
    prod = pd.DataFrame({
        "date": ["2022-01", "2022-06"],
        "country": ["Saudi Arabia", "Russia"],
        "production_mbd": [10.1, 9.8],
        "opec_member": ["Yes", "No"]
    })
    merged = combine_price_and_production(prices, prod)
    missing_price = merged[merged["date"] == "2022-06"]["oil_price"]
    assert missing_price.isna().all(), "Missing price date should produce NaN"

def test_merge_events_countries_not_in_acled_get_zero():
    merged = combine_price_and_production(make_oil_prices(), make_oil_prod())
    result = merge_events(merged, make_acled())
    row = result[(result["date"] == "2022-01") & (result["country"] == "Russia")]
    assert row["event_count"].values[0] == 0.0



