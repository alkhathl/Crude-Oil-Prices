import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summary_stats(df, column):

    print(df[column].describe())


def plot_oil_price_time_series(oil_prices):
    df = oil_prices.copy()

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

    df = df.groupby('date').agg({
        'oil_price': 'mean',
        'event_count': 'sum'
    }).reset_index()

    df = df.sort_values('date')

    df['date_str'] = df['date'].dt.strftime('%m-%y')

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax1.plot(df['date_str'], df['oil_price'], color='blue', linewidth=2)
    ax1.set_ylabel("Oil Price (USD)", color='blue')

    ax2 = ax1.twinx()
    ax2.plot(df['date_str'], df['event_count'], color='red', linewidth=2)
    ax2.set_ylabel("Political Events", color='red')

    ax1.set_title("Oil Price vs Political Violence Over Time")
    ax1.set_xlabel("Month-Year")

    step = max(1, len(df)//12)
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels(df['date_str'][::step], rotation=45)

    plt.tight_layout()
    plt.show()


def plot_event_map(detailed_events_df):
    import plotly.express as px

    df = detailed_events_df.copy()
    df = df[df['disorder_type'] == 'Political violence']

    country_counts = (
        df.groupby('country')
        .size()
        .reset_index(name='event_count')
    )

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


def plot_events_by_year_stacked(detailed_events_df):
    df = detailed_events_df.copy()
    df = df[df['disorder_type'] == 'Political violence']

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    grouped = (
        df.groupby(['year', 'event_type'])
        .size()
        .reset_index(name='count')
    )

    pivot = grouped.pivot(index='year', columns='event_type', values='count').fillna(0)

    pivot.plot(kind='bar', stacked=True, figsize=(12,6))

    plt.title("Political Violence by Year (Event Type Breakdown)")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=0)

    plt.legend(title="Event Type", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()