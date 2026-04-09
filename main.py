import os
import pandas as pd
from config import DATA_DIR, RESULTS_DIR
from load import get_oil_prices, get_oil_production, load_acled_data
from analyze import (
    plot_oil_price_time_series,
    plot_country_production,
    plot_events_vs_price_trend,
    plot_event_map,
    plot_events_by_year_stacked,
    summary_stats
)
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


    print("Fetching oil prices...")
    oil_prices = get_oil_prices()
    print("Oil prices preview:")
    print(oil_prices.head(5))

    print("Fetching oil production...")
    oil_prod = get_oil_production()
    print("Oil production preview:")
    print(oil_prod.head(5))


    merged_df = pd.merge(
        oil_prod,
        oil_prices,
        on="date",
        how="left"
    )


    merged_df["production_mbd"] = pd.to_numeric(merged_df["production_mbd"], errors="coerce")
    merged_df["oil_price"] = pd.to_numeric(merged_df["oil_price"], errors="coerce")

    print("\nMerged oil price + production (first 20 rows):")
    print(merged_df[["date", "country", "production_mbd", "oil_price"]].head(20))


    print("\nLoading ACLED data...")
    acled_df = load_acled_data(DATA_DIR)  # loads all files in DATA_DIR
    print(f"ACLED data loaded. Total rows: {len(acled_df)}")
    print("ACLED preview (first 20 rows):")
    print(acled_df.head(20))


    event_counts = (
        acled_df
        .groupby(['date', 'country'])
        .size()
        .reset_index(name='event_count')
    )


    merged_df = pd.merge(
        merged_df,
        event_counts,
        on=['date', 'country'],
        how='left'
    )
    merged_df['event_count'] = merged_df['event_count'].fillna(0)

    print("\nMerged dataset with event counts (first 20 rows):")
    print(merged_df.head(20))


    summary_df = merged_df.groupby('date', as_index=False).agg(
        oil_price=('oil_price', 'mean'),
        avg_production_mbd=('production_mbd', 'mean'),
        mean_production_mbd=('production_mbd', 'mean'),
        max_production_mbd=('production_mbd', 'max'),
        min_production_mbd=('production_mbd', 'min'),
        event_count=('event_count', 'sum')
    )
    summary_df['summary_id'] = range(1, len(summary_df)+1)
    print("\nMonthly summary table (first 20 rows):")
    print(summary_df.head(20))


    detailed_production_df = pd.merge(
        merged_df,
        summary_df[['summary_id', 'date']],
        on='date',
        how='left'
    )
    detailed_production_df['production_id'] = range(1, len(detailed_production_df)+1)
    print("\nDetailed production table preview (first 20 rows):")
    print(detailed_production_df.head(20))


    detailed_events_df = pd.merge(
        acled_df,
        summary_df[['summary_id', 'date']],
        on='date',
        how='left'
    )
    detailed_events_df = pd.merge(
        detailed_events_df,
        detailed_production_df[['production_id', 'summary_id', 'country']],
        on=['summary_id', 'country'],
        how='left'
    )
    detailed_events_df['event_id'] = range(1, len(detailed_events_df)+1)
    print("\nDetailed ACLED events table preview (first 20 rows):")
    print(detailed_events_df.head(20))

  
    print("\nSummary stats for oil_price:")
    summary_stats(merged_df, column="oil_price")
    print("\nSummary stats for production_mbd:")
    summary_stats(merged_df, column="production_mbd")

    print("\nSummary stats for oil_price:")
    summary_stats(merged_df, column="oil_price")
    print("\nSummary stats for production_mbd:")
    summary_stats(merged_df, column="production_mbd")

    plot_oil_price_time_series(oil_prices)
    plot_country_production(oil_prod)
    plot_events_vs_price_trend(merged_df)
    plot_event_map(detailed_events_df)
    plot_events_by_year_stacked(detailed_events_df)

