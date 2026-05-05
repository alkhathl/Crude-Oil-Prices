from pathlib import Path
from dotenv import load_dotenv
import os

#API

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("EIA_API_KEY")

# File Paths

DATA_DIR = Path(r"C:\Users\viiz\Documents\Spring 26`\510\Jupyter\Crude Oik\data")
RESULTS_DIR = "results"

OIL_PRICE_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
OIL_PRODUCTION_URL = "https://api.eia.gov/v2/international/data/"

# I had to download the ACLED conflict data in 5 batches as its covering over 1.2 million daily records
# batch download was too large and kept failing

ACLED_FILES = [
    "ACLED Data_10-13.csv",   # 2010 to 2013
    "ACLED Data_14-15.csv",   # 2014 to 2015
    "ACLED Data_2016.csv",    # 2016
    "ACLED Data_17-20.csv",   # 2017 to 2020
    "ACLED Data_21-25.csv",   # 2021 to 2025
]


# These are the 7 major oil producing countries I focused on
# The EIA API uses ISO country codes as identifiers,so I mapped each code to a readable country name
# I chose these 7 because together they account for 60% of global oil production

COUNTRIES = {
    "SAU": "Saudi Arabia",
    "RUS": "Russia",
    "IRQ": "Iraq",
    "ARE": "United Arab Emirates",
    "USA": "United States",
    "IRN": "Iran",
    "KWT": "Kuwait"
}

# This is the list of OPEC member countries from the 7 countries i included
# I use it to calculate OPEC's share of total production each month

OPEC_MEMBERS = {"Saudi Arabia", "Iraq", "United Arab Emirates", "Iran","Kuwait"}

# Key events to annotate on the premium chart
PREMIUM_EVENTS = {
    "Arab Spring": "2011-02-01",
    "Oil Glut":    "2015-07-01",
    "COVID":       "2020-04-01",
    "Ukraine War": "2022-03-01",
}

# These are the 5 major geopolitical events I selected for the before/after price impact analysis.
# I chose events that based on the following criteria:
# 1. Had a clear start date
# 2. Were widely recognized as geopolitically significant
# 3. Involved countries and regions relevant to oil supply
# The dates here represent when each event started or escalated significantly

MAJOR_EVENTS = {
    "Arab Spring":         "2011-01-01",  # Libyan civil war disrupted production
    "ISIS Iraq Offensive": "2014-06-01",  # ISIS captured Mosul, threatened Iraqi oil fields
    "COVID Crash":         "2020-03-01",  # Global demand collapsed as lockdowns began
    "Russia-Ukraine War":  "2022-02-01",  # Russia invaded Ukraine, triggering sanctions
    "Hamas-Israel":        "2023-10-01",  # Hamas attack on Israel, regional escalation fears
}

# Features for the baseline model
# Price lags and rolling average to track price trends
# Production levels and lags to track supply trends
# OPEC share and production shock to capture supply changes

PROD_FEATURES = [
    "price_lag1",            # oil price 1 month ago
    "price_lag2",            # oil price 2 months ago
    "price_lag3",            # oil price 3 months ago
    "price_roll3",           # average oil price over the past 3 months
    "total_production_mbd",  # total production across all 7 countries this month
    "avg_production_mbd",    # average production per country this month
    "opec_production",       # total production from OPEC members only
    "opec_share",            # OPEC production as a % of total production
    "production_lag1",       # total production 1 month ago
    "production_lag2",       # total production 2 months ago
    "production_lag3",       # total production 3 months ago
    "production_roll3",      # average production over the past 3 months
    "production_shock",      # how much production changed vs 3 months ago
]
