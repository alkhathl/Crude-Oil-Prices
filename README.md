\# Geopolitical Impact on Crude Oil Prices



\## Introduction

Oil prices are highly sensitive to geopolitical events that disrupt supply, threaten trade

routes, and shift market expectations. This project examines how major geopolitical

developments including conflicts and sanctions, impact global crude oil prices using 15

years of monthly data from 2010 to 2024.This analysis investigates whether and how

conflict events correlate with price movements and quantifies the geopolitical risk

premium markets price above supply fundamentals. The objective is to understand when

and why political instability drives oil price volatility and when it does not.



\## Data Sources



| # | Source | Description | Type | Size |

|---|--------|-------------|------|------|

| 1 | EIA API — Brent Crude Prices | Monthly Brent crude spot prices | REST API | 180 rows |

| 2 | EIA API — International Production | Monthly production by country for 7 major producers | REST API | 1,260 rows |

| 3 | ACLED CSV Files | Political violence events globally 2010–2024 | CSV (6 files) | 1.2M rows |



\## Analysis

The project is structured in three parts:

1\. Exploratory: time series of oil prices, production by country,

&#x20;  political violence by year and geography

2\. Impact analysis:  oil price 3 months before vs after 5 major

&#x20;  geopolitical events to measure direct price impact

3\. Geopolitical premium: a supply baseline model trained on

&#x20;  production data only is used to isolate the geopolitical risk

&#x20;  premium above supply fundamentals



\## Summary of Results

\- Supply-disrupting conflicts raise prices significantly:

&#x20; Arab Spring +31.7%, Russia-Ukraine War +38.7%

\- Contained conflicts lower prices: Hamas-Israel -7.4%, ISIS Iraq -5.9%

\- The average geopolitical premium across 169 months is $0.56/barrel

\- During the Ukraine War the average premium was $4.03/barrel above

&#x20; supply fundamentals



\## How to Run



\### Requirements



\### Step 1 — Install dependencies

```

pip install -r requirements.txt

```



\### Step 2 — Set up your API key

This project requires a free EIA API key.

1\. Register at https://www.eia.gov/opendata/

2\. Create a `.env` file in the project root folder

3\. Add your key exactly as shown in `.env.example`:

```

EIA\_API\_KEY=your\_api\_key\_here

```



\### Step 3 — Add ACLED data files

Download the ACLED CSV files and place them in the `data/` folder.

The required filenames are listed in `src/config.py` under `ACLED\_FILES`.



\### Step 4 — Run the analysis

From the project root folder run:

```

python main.py

```



All charts will be saved automatically to the `results/` folder.



\### Step 5 — Run the tests

```

pytest tests.py -v

```



\## AI Disclosure

The `build\_baseline\_model` function in `src/analyze.py` was implemented

using scikit-learn's Random Forest Regressor with guidance from

ChatGPT . All AI-generated code sections are labeled

with `# AI generated:` comments in the source files.

