# AI DISCLOSURE: The baseline supply model used to calculate the geopolitical
# premium (build_baseline_model function) I implemented using scikit-learn's
# Random Forest Regressor with guidance from an AI assistant(ChatGPT).

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from config import MAJOR_EVENTS, PREMIUM_EVENTS, PROD_FEATURES, RESULTS_DIR


def summary_stats(df, column):

    print(df[column].describe())

def plot_oil_price_time_series(oil_prices):
    df = oil_prices.copy()

    # I'm convert date column to datetime so matplotlib can plot it correctly
    df['date'] = pd.to_datetime(df['date'])
    df['oil_price'] = pd.to_numeric(df['oil_price'], errors='coerce')
    df = df.sort_values('date')

    plt.figure(figsize=(12, 6))

    plt.plot(
        df['date'],
        df['oil_price'],
        linewidth=2,
        marker='o',
        markersize=3
    )

    plt.title("Monthly Oil Prices (USD per barrel)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Oil Price (USD)")
    step = max(1, len(df) // 12)
    plt.xticks(df['date'][::step], rotation=45)
    plt.locator_params(axis='y', nbins=6)
    # I set the y-axis limits with a small buffer above and below
    # so the line does not touch the edges of the chart
    plt.ylim(df['oil_price'].min() * 0.95, df['oil_price'].max() * 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_country_production(oil_prod):
    df = oil_prod.copy()
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x='date', y='production_mbd', hue='country', linewidth=2)
    plt.title("Monthly Oil Production by Country (mb/d)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Production (million barrels/day)")
    step = max(1, len(df)//12)
    plt.xticks(df['date'][::step])
    plt.xticks(rotation=45)
    plt.legend(title='Country')
    plt.tight_layout()
    plt.show()


def plot_events_vs_price_trend(merged_df):
    df = merged_df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Aggregate to one row per month, average price and total events
    df = df.groupby('date').agg({
        'oil_price': 'mean',
        'event_count': 'sum'
    }).reset_index()
    df = df.sort_values('date')
    # I'm formatting dates as month-year strings for the x-axis labels
    df['date_str'] = df['date'].dt.strftime('%m-%y')

    fig, ax1 = plt.subplots(figsize=(12,6))

    # I'm plotting oil price on the left y-axis
    ax1.plot(df['date_str'], df['oil_price'], color='blue', linewidth=2)
    ax1.set_ylabel("Oil Price (USD)", color='blue')

    # I'm using twinx() to create a second axis for the events count
    ax2 = ax1.twinx()
    ax2.plot(df['date_str'], df['event_count'], color='red', linewidth=2)
    ax2.set_ylabel("Political Events", color='red')
    ax1.set_title("Oil Price vs Political Violence Over Time")
    ax1.set_xlabel("Month-Year")
    # I set it to only show every 12th label as it was overcrowded in the x-axis and unreadable
    step = max(1, len(df)//12)
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels(df['date_str'][::step], rotation=45)
    plt.tight_layout()
    plt.show()


def plot_event_map(acled_df):
    # I'm counting total events per country across all years
    country_counts = acled_df.groupby("country").size().reset_index(name="event_count")

    # I used plotly's scatter_geo to create an interactive bubble map
    # I set locationmode="country names" so that I can use country name strings
    # directly without needing to convert them to ISO codes

    fig = px.scatter_geo(
        country_counts,
        locations="country",
        locationmode="country names",
        size="event_count",
        color="event_count",
        title="Political Violence by Country",
        projection="natural earth"
    )
    fig.show()


def plot_events_by_year_stacked(acled_df):
    df = acled_df.copy()
    df = df[df['disorder_type'] == 'Political violence']
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    # I'm countting events grouped by year and event type
    grouped = (
        df.groupby(['year', 'event_type'])
        .size()
        .reset_index(name='count')
    )

    # Pivot so each event type becomes its own column to create a stacked bar chart
    pivot = grouped.pivot(index='year', columns='event_type', values='count').fillna(0)
    pivot.plot(kind='bar', stacked=True, figsize=(12,6))
    plt.title("Political Violence by Year (Event Type Breakdown)")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=0)
    plt.legend(title="Event Type", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def plot_conflict_impact_windows(monthly_df):

    df = monthly_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # I set date as the index so I can slice by date range easily
    df = df.set_index("date").sort_index()

    results = []
    for event_name, event_date in MAJOR_EVENTS.items():
        edate  = pd.Timestamp(event_date)

        # I'm calculating average price in the 3 months before the event
        before = df.loc[
            edate - pd.DateOffset(months=3) : edate - pd.DateOffset(months=1),
            "oil_price"
        ].mean()

        # I'm calculating average price in the 3 months after the event
        after = df.loc[
            edate + pd.DateOffset(months=1) : edate + pd.DateOffset(months=3),
            "oil_price"
        ].mean()

        # I set it to show only events that both before and after values exist
        if pd.notna(before) and pd.notna(after):
            change_pct = round(((after - before) / before) * 100, 1)
            results.append({
                "Event":                event_name,
                "Avg Price Before ($)": round(before, 2),
                "Avg Price After ($)":  round(after,  2),
                "Price Change (%)":     change_pct,
            })

    impact_df = pd.DataFrame(results)
    print("\nPrice impact around major geopolitical events:")
    print(impact_df.to_string(index=False))

    # I'm building the grouped bar chart
    x     = range(len(impact_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - width/2 for i in x], impact_df["Avg Price Before ($)"],
           width, label="Avg Price 3mo Before", color="steelblue")
    ax.bar([i + width/2 for i in x], impact_df["Avg Price After ($)"],
           width, label="Avg Price 3mo After", color="coral")

    # I'm annotating each after bar with the percentage change.
    # Green for price increases, red for price decreases
    for i, row in impact_df.iterrows():
        color = "green" if row["Price Change (%)"] > 0 else "red"
        ax.annotate(
            f"{row['Price Change (%)']}%",
            xy=(i + width/2, row["Avg Price After ($)"] + 1),
            ha="center", fontsize=10, color=color, fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(impact_df["Event"], rotation=20, ha="right")
    ax.set_ylabel("Oil Price (USD per barrel)")
    ax.set_title("Oil Price Before vs After Major Geopolitical Events", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart5_conflict_impact.png"))
    plt.show()
    return impact_df


# DISCLOSURE: build_baseline_model I implemented using scikit-learn's
# Random Forest Regressor with guidance from an AI assistant (ChatGPT)

def build_baseline_model(ext_df):
    """
    Trains a supply baseline model using only production features
    no conflict data. The model predicts what oil price should be
    based purely on how much oil is being produced.

    The difference between actual price and this predicted baseline
    is the geopolitical premium: the extra amount markets charge
    above supply fundamentals due to political risk.

    DISCLOSURE: This function was implemented with guidance
    from an AI assistant (ChatGPT)

    Parameters:
        ext_df : extended monthly dataset from build_extended_monthly()

    Returns:
        model      : the trained Random Forest model
        df         : ext_df with two new columns added:
                     baseline_price and geo_premium
    """

    X = ext_df[PROD_FEATURES]
    y = ext_df["oil_price"]

    split   = ext_df[ext_df["date"] < "2022-01-01"].shape[0]
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test  = X.iloc[split:]
    y_test  = y.iloc[split:]

    print("\n--- Baseline Production Model ---")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)

    print(f"Test MAE : {mean_absolute_error(y_test, test_preds):.2f}")
    print(f"Test R²  : {r2_score(y_test, test_preds):.3f}")

    df = ext_df.copy()
    df["baseline_price"] = model.predict(X)
    df["geo_premium"]    = df["oil_price"] - df["baseline_price"]
    return model, df


def plot_geopolitical_premium(premium_df):

    # I'm filtering to 2010 onwards as this is where ACLED conflict data is available
    # so the premium chart aligns with the rest of my analysis

    df = premium_df[premium_df["date"] >= "2010-01-01"].copy()

    # I'm printing a summary of the premium for key periods
    print("\nGeopolitical Premium Summary (2010-2024):")
    print(f"  All months avg premium      : ${df['geo_premium'].mean():.2f}")
    print(f"  2011 Arab Spring            : ${df[df['date'].dt.year == 2011]['geo_premium'].mean():.2f}")
    print(f"  2020 COVID                  : ${df[df['date'].dt.year == 2020]['geo_premium'].mean():.2f}")
    print(f"  2022 Ukraine War            : ${df[df['date'].dt.year == 2022]['geo_premium'].mean():.2f}")

    # I'm using sharex=True to link both panels to the same x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top panel: actual price vs supply baseline
    ax1.plot(df["date"], df["oil_price"],
             label="Actual Price", linewidth=2, color="steelblue")
    ax1.plot(df["date"], df["baseline_price"],
             label="Supply Baseline", linewidth=2, color="gray", linestyle="--")

    # Red shading is showing when actual price is above baseline (geopolitical premium)
    ax1.fill_between(
        df["date"], df["baseline_price"], df["oil_price"],
        where=(df["geo_premium"] > 0),
        alpha=0.3, color="red", label="Geopolitical Premium (+)"
    )
    # Blue shading is showing when actual price is below baseline (geopolitical discount)
    ax1.fill_between(
        df["date"], df["baseline_price"], df["oil_price"],
        where=(df["geo_premium"] < 0),
        alpha=0.3, color="blue", label="Geopolitical Discount (-)"
    )
    ax1.set_ylabel("Oil Price (USD)")
    ax1.set_title(
        "Actual Oil Price vs Supply Baseline\n"
        "(Red = premium above fundamentals, Blue = discount below)"
    )
    ax1.legend()


    # Bottom panel: monthly premium as bars
    # Red bar = premium (actual above baseline), blue = discount
    bar_colors = ["red" if x > 0 else "steelblue" for x in df["geo_premium"]]
    ax2.bar(df["date"], df["geo_premium"], color=bar_colors, alpha=0.7, width=25)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Geopolitical Premium (USD/barrel)")
    ax2.set_xlabel("Date")
    ax2.set_title("Monthly Geopolitical Risk Premium in Oil Price")

    # I'm adding vertical dashed lines and labels for key events
    for label, date in PREMIUM_EVENTS.items():
        edate = pd.Timestamp(date)
        if df["date"].min() <= edate <= df["date"].max():
            ax2.axvline(edate, color="orange", linestyle="--", alpha=0.8)
            ax2.text(edate, df["geo_premium"].max() * 0.85,
                     label, fontsize=8, rotation=45, color="darkorange")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart7_geopolitical_premium.png"))
    plt.show()