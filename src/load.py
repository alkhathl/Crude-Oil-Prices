import requests
import pandas as pd
from config import API_KEY, COUNTRIES, OPEC_MEMBERS, OIL_PRICE_URL, OIL_PRODUCTION_URL, ACLED_FILES
from pathlib import Path


def get_oil_prices(start="2010-01", end="2024-12") -> pd.DataFrame:

    # I set the length=5000 to ensures I get all months in one request
    params = {
        "api_key": API_KEY,
        "frequency": "monthly",
        "data[]": "value",
        "facets[series][]": "RBRTE", # RBRTE is the EIA series code for Brent crude
        "start": start,
        "end": end,
        "length": 5000
    }
    resp = requests.get(OIL_PRICE_URL, params=params, timeout=30).json()
    if "response" not in resp:
        raise RuntimeError(f"EIA API error: {resp}")
    # Extracting just the period and value columns from the response
    # and rename them to date and oil_price
    df = pd.DataFrame(resp["response"]["data"])[["period", "value"]]
    df.columns      = ["date", "oil_price"]
    # I'm convert oil_price to number. I added errors="coerce" so any value
    # that cannot be converted for example  missing data becomes NaN
    df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")
    return df


def get_oil_production(start="2010-01", end="2024-04"):
    # I will collect each country's dataframe in this list
    # and combine them all at the end

    all_dfs = []

    # Each country needs its own API request.
    # activityId=1 means crude oil production
    # productId=53 means total oil
    # unit=TBPD means thousand barrels per day

    for iso_code, country_name in COUNTRIES.items():
        params = {
            "api_key": API_KEY,
            "frequency": "monthly",
            "data[]": "value",
            "facets[activityId][]": "1",
            "facets[productId][]": "53",
            "facets[countryRegionId][]": iso_code,
            "facets[unit][]": "TBPD",
            "start": start,
            "end": end,
            "length": 500
        }

        resp = requests.get(OIL_PRODUCTION_URL, params=params)
        data = resp.json()
        rows = data.get("response", {}).get("data", [])
        # If no data came back for this country, skip it and print a warning
        # so I know which countries had issues without crashing the whole run
        if not rows:
            print(f"No data for {country_name}")
            continue
        # I'm adding the country name as a column so I know which country
        # each row belongs to after combining all countries together
        df = pd.DataFrame(rows)
        df["country"] = country_name
        all_dfs.append(df)
    # Stack all country dataframes into one big dataframe
    df_all = pd.concat(all_dfs, ignore_index=True)
    # I'm only keeping the columns i need and renamed them to practical names
    df_all = df_all[["period", "country", "value", "unit"]].copy()
    df_all.columns = ["date", "country", "production_tbpd", "unit"]
    # i'm chainging the data type for production values to numbers
    df_all["production_tbpd"] = pd.to_numeric(df_all["production_tbpd"], errors="coerce")
    # Converting from thousand barrels per day to million barrels per day
    df_all["production_mbd"] = (df_all["production_tbpd"] / 1000).round(3)
    #I'm flagging all OPEC members
    df_all["opec_member"] = df_all["country"].apply(lambda x: "Yes" if x in OPEC_MEMBERS else "No")
    # Sorting by date then country so the data is in a consistent order
    df_all = df_all.sort_values(["date", "country"]).reset_index(drop=True)
    return df_all


# Combine all 5 files into one dataframe

def load_acled_data(data_dir):

    DATA_DIR = Path(data_dir)
    acled_dfs = []

    for f in ACLED_FILES:
        path = DATA_DIR / f
        print(f"Loading {path}...")
        df = pd.read_csv(path, low_memory=False)
        # I filtered to political violence only
        df = df[df["disorder_type"] == "Political violence"].copy()
        # Converting the event date to datetime format so I can extract the year and month
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        # I'm merging to monthly
        df['date'] = df['event_date'].dt.to_period('M').astype(str)
        # Removing any extra spaces from country names as I faced merging issues
        df['country'] = df['country'].str.strip()
        # I'm keep only the columns I need for the analysis
        df = df[["date", "country", "event_type", "sub_event_type", "disorder_type"]]
        acled_dfs.append(df)

    acled_df = pd.concat(acled_dfs, ignore_index=True)

    print("ACLED data loaded. Number of rows:", len(acled_df))
    return acled_df