import os
from config import DATA_DIR, RESULTS_DIR
from load import get_oil_prices, get_oil_production, load_acled_data
from process import merge_price_production_events, build_monthly_features, build_extended_monthly
from analyze import (
    plot_oil_price_time_series,
    plot_country_production,
    plot_events_vs_price_trend,
    plot_event_map,
    plot_events_by_year_stacked,
    summary_stats,
    plot_conflict_impact_windows,
    build_baseline_model,
    plot_geopolitical_premium,
)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # I loaded two separate date ranges:
    # 2010 to 2024 for the geopolitical analysis (as ACLED is only available 2010-2024)
    # 000 to 2024 for the baseline model (to feed/train the model more datapoints)

    print("Fetching oil prices (2010-2024)")
    oil_prices = get_oil_prices(start="2010-01")

    print("Fetching oil production (2010-2024)")
    oil_prod = get_oil_production(start="2010-01")

    print("Loading ACLED conflict data")
    acled_df = load_acled_data(DATA_DIR)

    # Extended data for the baseline model only
    # extra 10 years of history helps the model learn the
    # relationship between production and price more accurately

    print("Fetching extended oil prices (2000-2024) for baseline model")
    oil_prices_ext = get_oil_prices(start="2000-01")

    print("Fetching extended oil production (2000-2024) for baseline model")
    oil_prod_ext = get_oil_production(start="2000-01")

    print("\nMerging data sources...")
    merged_df  = merge_price_production_events(oil_prices, oil_prod, acled_df)
    monthly_df = build_monthly_features(merged_df)
    ext_df     = build_extended_monthly(oil_prices_ext, oil_prod_ext)

    print(f"Monthly dataset : {len(monthly_df)} rows")
    print(f"Extended dataset: {len(ext_df)} rows")

    # Quick summary stats to sanity check the data before running charts
    # I added these to make sure the price and production ranges look
    # reasonable before spending time generating all the charts

    print("\nSummary stats for oil_price:")
    summary_stats(merged_df, column="oil_price")
    print("\nSummary stats for production_mbd:")
    summary_stats(merged_df, column="production_mbd")

    # Initaial analysis stage and gives us the status quo before digging deeper on the cause of the price change
    print("\nEXPLORATORY ANALYSIS")
    plot_oil_price_time_series(oil_prices)
    plot_country_production(oil_prod)
    plot_events_vs_price_trend(merged_df)
    plot_event_map(acled_df)
    plot_events_by_year_stacked(acled_df)

    print("\n IMPACT & CORRELATION")
    plot_conflict_impact_windows(monthly_df)
    model, premium_df = build_baseline_model(ext_df)
    plot_geopolitical_premium(premium_df)
