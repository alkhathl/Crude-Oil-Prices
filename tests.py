# AI DISCLOSURE:
# The ACLED dataset alone has over 1.2 million rows and after merging
# all three data sources together the total records exceeded 10 million
# At that scale I was concerned any data quality issue or API not fetching all the records will wreck my entire analysis
# I have used AI to help generate a thorough test on all aspect that i was concerned about
# I gave it a specific parameters/areas i wanted to test
# The specific things I asked to be tested as follow:
# API tests:
#   - get_oil_prices() returns a non-empty dataframe with date and oil_price columns
#   - get_oil_production() returns a non-empty dataframe with date, country,
#     production_mbd and opec_member columns
#   - All 7 countries from config.py are present in the production data
#   - All oil_price values are numeric and positive
#   - All production_mbd values are numeric and positive
#
# Merge tests:
#   - Merged dataframe has all required columns: date, country,
#     production_mbd, oil_price, event_count
#   - Row count after merge matches production row count
#   - oil_price and production_mbd are stored as float not string
#   - event_count has no null values after merge
#   - Saudi Arabia January 2022 has exactly 1 event (matching sample data)
#   - Russia January 2022 has 0 events (no Russia events in sample data)
#   - Production rows with no matching price date produce NaN oil_price
#     rather than crashing

import pandas as pd
import pytest
from process import merge_price_production_events


def make_oil_prices():
    return pd.DataFrame({
        "date":      ["2022-01", "2022-02", "2022-03"],
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
        "event_type":     ["Battles", "Battles",
                           "Explosions/Remote violence", "Protests"],
        "sub_event_type": ["Armed clash", "Armed clash",
                           "Air/drone strike", "Peaceful protest"],
        "disorder_type":  ["Political violence", "Political violence",
                           "Political violence", "Political violence"]
    })


#API tests

def test_get_oil_prices_returns_dataframe():
    from load import get_oil_prices
    df = get_oil_prices()
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert not df.empty,                 "DataFrame should not be empty"
    assert "date"      in df.columns,    "Missing 'date' column"
    assert "oil_price" in df.columns,    "Missing 'oil_price' column"

def test_get_oil_production_returns_dataframe():
    from load import get_oil_production
    df = get_oil_production()
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert not df.empty,                 "DataFrame should not be empty"
    for col in ["date", "country", "production_mbd", "opec_member"]:
        assert col in df.columns, f"Missing column: {col}"

def test_get_oil_production_has_expected_countries():
    from load import get_oil_production
    from config import COUNTRIES
    df       = get_oil_production()
    expected = set(COUNTRIES.values())
    actual   = set(df["country"].unique())
    missing  = expected - actual
    assert not missing, f"Missing countries in production data: {missing}"

def test_oil_price_values_are_numeric():
    from load import get_oil_prices
    df = get_oil_prices()
    df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")
    assert df["oil_price"].notna().all(), "oil_price contains null values"
    assert (df["oil_price"] > 0).all(),   "oil_price should always be positive"

def test_production_values_are_numeric():
    from load import get_oil_production
    df = get_oil_production()
    assert df["production_mbd"].notna().all(), "production_mbd contains nulls"
    assert (df["production_mbd"] > 0).all(),   "production_mbd should be positive"


# Merge tests

def test_merge_has_required_columns():
    result = merge_price_production_events(
        make_oil_prices(), make_oil_prod(), make_acled()
    )
    for col in ["date", "country", "production_mbd", "oil_price", "event_count"]:
        assert col in result.columns, f"Missing column after merge: {col}"

def test_merge_row_count_matches_production():
    result = merge_price_production_events(
        make_oil_prices(), make_oil_prod(), make_acled()
    )
    assert len(result) == len(make_oil_prod()), \
        "Row count after merge should match production rows"

def test_merge_column_types_are_numeric():
    result = merge_price_production_events(
        make_oil_prices(), make_oil_prod(), make_acled()
    )
    assert pd.api.types.is_float_dtype(result["oil_price"]), \
        "oil_price should be float"
    assert pd.api.types.is_float_dtype(result["production_mbd"]), \
        "production_mbd should be float"

def test_merge_event_count_no_nulls():
    result = merge_price_production_events(
        make_oil_prices(), make_oil_prod(), make_acled()
    )
    assert result["event_count"].notna().all(), \
        "event_count should not have null values"

def test_merge_event_count_correct_value():
    result = merge_price_production_events(
        make_oil_prices(), make_oil_prod(), make_acled()
    )
    row = result[
        (result["date"] == "2022-01") & (result["country"] == "Saudi Arabia")
    ]
    assert row["event_count"].values[0] == 1.0, \
        "Saudi Arabia Jan 2022 should have exactly 1 event"

def test_merge_country_with_no_events_gets_zero():
    result = merge_price_production_events(
        make_oil_prices(), make_oil_prod(), make_acled()
    )
    row = result[
        (result["date"] == "2022-01") & (result["country"] == "Russia")
    ]
    assert row["event_count"].values[0] == 0.0, \
        "Russia Jan 2022 should have 0 events"

def test_merge_handles_missing_price_date():
    prices = pd.DataFrame({
        "date":      ["2022-01"],
        "oil_price": [77.5]
    })
    prod = pd.DataFrame({
        "date":           ["2022-01", "2022-06"],
        "country":        ["Saudi Arabia", "Russia"],
        "production_mbd": [10.1, 9.8],
        "opec_member":    ["Yes", "No"]
    })
    result = merge_price_production_events(prices, prod, make_acled())
    missing_price = result[result["date"] == "2022-06"]["oil_price"]
    assert missing_price.isna().all(), \
        "Months with no price data should have NaN oil_price"