import pandas as pd

def merge_price_production_events(oil_prices, oil_prod,acled_df):
    # Merging production with prices
    df = pd.merge(oil_prod, oil_prices, on="date", how="left")
    # I'm converting both numeric columns into numbers just in case it comes as a string
    df["production_mbd"] = pd.to_numeric(df["production_mbd"], errors="coerce")
    df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")

    # Counting how many conflict events happened per country
    event_counts = (
        acled_df
        .groupby(["date", "country"])
        .size()
        .reset_index(name="event_count")
    )
    # Joining the event counts to the main dataframe
    df = pd.merge(df, event_counts, on=["date", "country"], how="left")
    df["event_count"] = df["event_count"].fillna(0)
    return df

def build_summary_table(merged_df):
    # I built a summary table initally to assess the general stats in production max and average value
    summary_df = merged_df.groupby("date", as_index=False).agg(
        oil_price=("oil_price", "mean"),
        avg_production_mbd=("production_mbd", "mean"),
        max_production_mbd=("production_mbd", "max"),
        min_production_mbd=("production_mbd", "min"),
        event_count=("event_count", "sum")
    )
    summary_df["summary_id"] = range(1, len(summary_df) + 1)
    return summary_df


def build_detailed_production(merged_df, summary_df):
    # I built detailed tables initally for both production and events
    # to investigate and find patterns before deciding on the analysis
    df = pd.merge(merged_df, summary_df[["summary_id", "date"]], on="date", how="left")
    df["production_id"] = range(1, len(df) + 1)
    return df

def build_detailed_events(acled_df, summary_df, detailed_production_df):
    df = pd.merge(acled_df, summary_df[["summary_id", "date"]], on="date", how="left")
    df = pd.merge(df, detailed_production_df[["production_id", "summary_id", "country"]],
                  on=["summary_id", "country"], how="left")
    df["event_id"] = range(1, len(df) + 1)
    return df

def build_monthly_features(merged_df):

    df = merged_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # I'm averaging oil_price across all 7 countries as it the same global price
    # I'm summing all production to get the total across all 7 countries
    # I'm summing event_count the global monthly total

    monthly = df.groupby("date", as_index=False).agg(
        oil_price            = ("oil_price",      "mean"),
        total_production_mbd = ("production_mbd", "sum"),
        avg_production_mbd   = ("production_mbd", "mean"),
        total_event_count    = ("event_count",     "sum"),
    )

    monthly = monthly.sort_values("date").reset_index(drop=True)

    # I'm calculating OPEC production separately and summing their production per month
    opec_df= df[df["opec_member"] == "Yes"]
    opec_monthly= opec_df.groupby("date")["production_mbd"].sum().reset_index()
    opec_monthly.columns = ["date", "opec_production"]
    monthly = pd.merge(monthly, opec_monthly, on="date", how="left")

    # I created lag features for price, production, and events
    # going back 1, 2, and 3 months using shift(1) moveing the column
    # down by one row so each month gets the value from the month before it

    for lag in [1, 2, 3]:
        monthly[f"price_lag{lag}"]      = monthly["oil_price"].shift(lag)
        monthly[f"production_lag{lag}"] = monthly["total_production_mbd"].shift(lag)
        monthly[f"event_lag{lag}"]      = monthly["total_event_count"].shift(lag)

    # I've created 3-month rolling averages
    # using rolling(3).mean() to calculate the average of the current month
    # and the 2 months before it to smooths out short-term spikes
    # For price_roll3 I added .shift(1) so it only uses past month's price

    monthly["event_roll3"]= monthly["total_event_count"].rolling(3).mean()
    monthly["production_roll3"] = monthly["total_production_mbd"].rolling(3).mean()
    monthly["price_roll3"]= monthly["oil_price"].rolling(3).mean().shift(1)

    # I'm measuring production shock by how much production changed
    # compared to 3 months ago as a percentage change

    monthly["production_shock"] = (
        (monthly["total_production_mbd"] - monthly["production_lag3"])
        / (monthly["production_lag3"])
    )
    # I'm calculating OPEC share

    monthly["opec_share"] = (
        monthly["opec_production"] / (monthly["total_production_mbd"])
    )

    # I'm dropping the first few rows as the lag column lag value is NaN
    # because there are no previous months to look back to yet.

    monthly = monthly.dropna().reset_index(drop=True)
    return monthly


def build_extended_monthly(oil_prices, oil_prod):
    # This is the same function as build_monthly_features
    # but i'm not including conflict data as it only goes back to 2010
    # The extra years from 2000 to 2009 give the baseline model more data to learn
    # the relationship between production and price

    monthly = oil_prod.groupby("date", as_index=False).agg(
        total_production_mbd = ("production_mbd", "sum"),
        avg_production_mbd   = ("production_mbd", "mean"),
    )

    opec_df= oil_prod[oil_prod["opec_member"] == "Yes"]
    opec_monthly= opec_df.groupby("date")["production_mbd"].sum().reset_index()
    opec_monthly.columns = ["date", "opec_production"]
    monthly = pd.merge(monthly, opec_monthly, on="date", how="left")

    # I'm merging  with prices using inner join to only keep months
    # where both production and price data are available
    df = pd.merge(monthly, oil_prices, on="date", how="inner")
    df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")

    # I'm convert date to datetime format so it works correctly
    # with the time-based train/test split in analyze.py
    df["date"]= pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for lag in [1, 2, 3]:
        df[f"price_lag{lag}"]= df["oil_price"].shift(lag)
        df[f"production_lag{lag}"]= df["total_production_mbd"].shift(lag)

    df["price_roll3"]= df["oil_price"].rolling(3).mean().shift(1)
    df["production_roll3"] = df["total_production_mbd"].rolling(3).mean()

    df["opec_share"]= df["opec_production"] / (df["total_production_mbd"] + 1e-9)
    df["production_shock"] = (
        (df["total_production_mbd"] - df["production_lag3"])
        / (df["production_lag3"] + 1e-9)
    )

    # I'm dropping rows that has NaN lag values
    df = df.dropna().reset_index(drop=True)
    return df


