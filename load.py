import requests
import pandas as pd
from config import API_KEY, COUNTRIES, OPEC_MEMBERS, OIL_PRICE_URL, OIL_PRODUCTION_URL


def get_oil_prices() -> pd.DataFrame:

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

    params = {
        "api_key": API_KEY,
        "frequency": "monthly",
        "data[]": "value",
        "facets[series][]": "RBRTE",
        "start": "2015-01",
        "end": "2024-12",
        "length": 5000
    }

    r = requests.get(url, params=params)
    resp = r.json()

    if "response" not in resp:
        raise RuntimeError(f"API Error: {resp}")

    df = pd.DataFrame(resp["response"]["data"])

    df = df[["period", "value"]]
    df.columns = ["date", "oil_price"]

    return df


def get_oil_production(start="2015-01", end="2024-04"):
    all_dfs = []

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
        if not rows:
            print(f"No data for {country_name}")
            continue

        df = pd.DataFrame(rows)
        df["country"] = country_name
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all[["period", "country", "value", "unit"]].copy()
    df_all.columns = ["date", "country", "production_tbpd", "unit"]
    df_all["production_tbpd"] = pd.to_numeric(df_all["production_tbpd"], errors="coerce")
    df_all["production_mbd"] = (df_all["production_tbpd"] / 1000).round(3)
    df_all["opec_member"] = df_all["country"].apply(lambda x: "Yes" if x in OPEC_MEMBERS else "No")
    df_all = df_all.sort_values(["date", "country"]).reset_index(drop=True)
    return df_all

def load_acled_data(data_dir):
    import pandas as pd
    from pathlib import Path

    DATA_DIR = Path(data_dir)

    files = [
        "ACLED Data_2015.csv",
        "ACLED Data_2016.csv",
        "ACLED Data_17-20.csv",
        "ACLED Data_21-25.csv"
    ]

    acled_dfs = []

    for f in files:
        path = DATA_DIR / f
        print(f"Loading {path}...")
        df = pd.read_csv(path, low_memory=False)

        df = df[df["disorder_type"] == "Political violence"].copy()


        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df['date'] = df['event_date'].dt.to_period('M').astype(str)

        df['country'] = df['country'].str.strip()

        df = df[["date", "country", "event_type", "sub_event_type", "disorder_type"]]

        acled_dfs.append(df)

    acled_raw = pd.concat(acled_dfs, ignore_index=True)

    print("ACLED data loaded. Number of rows:", len(acled_raw))
    return acled_raw