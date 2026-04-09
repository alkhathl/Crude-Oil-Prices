import pandas as pd

def process_acled_data(acled_df):
    acled_df["event_date"] = pd.to_datetime(acled_df["event_date"])
    acled_df["date"] = acled_df["event_date"].dt.to_period("M").astype(str)
    grouped = (
        acled_df
        .groupby(["date", "country"])
        .size()
        .reset_index(name="event_count")
    )
    return grouped

def combine_price_and_production(oil_prices, oil_prod):
    merged = pd.merge(oil_prod, oil_prices, on="date", how="left")
    merged["production_mbd"] = pd.to_numeric(merged["production_mbd"], errors="coerce")
    merged["oil_price"] = pd.to_numeric(merged["oil_price"], errors="coerce")
    return merged

def merge_events(merged_df, acled_df):
    event_counts = (
        acled_df
        .groupby(["date", "country"])
        .size()
        .reset_index(name="event_count")
    )
    merged = pd.merge(merged_df, event_counts, on=["date", "country"], how="left")
    merged["event_count"] = merged["event_count"].fillna(0)
    return merged

def build_summary_table(merged_df):
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
    df = pd.merge(merged_df, summary_df[["summary_id", "date"]], on="date", how="left")
    df["production_id"] = range(1, len(df) + 1)
    return df

def build_detailed_events(acled_df, summary_df, detailed_production_df):
    df = pd.merge(acled_df, summary_df[["summary_id", "date"]], on="date", how="left")
    df = pd.merge(df, detailed_production_df[["production_id", "summary_id", "country"]],
                  on=["summary_id", "country"], how="left")
    df["event_id"] = range(1, len(df) + 1)
    return df