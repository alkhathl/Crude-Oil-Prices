from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("EIA_API_KEY")

DATA_DIR = Path(r"C:\Users\viiz\Documents\Spring 26`\510\Jupyter\Crude Oik\data")
RESULTS_DIR = "results"

COUNTRIES = {
    "SAU": "Saudi Arabia",
    "RUS": "Russia",
    "IRQ": "Iraq",
    "ARE": "United Arab Emirates",
    "USA": "United States",
    "IRN": "Iran",
    "KWT": "Kuwait"
}

OPEC_MEMBERS = {"Saudi Arabia", "Iraq", "United Arab Emirates", "Iran","Kuwait"}


OIL_PRICE_URL = "https://api.eia.gov/v2/series/data/"
OIL_PRODUCTION_URL = "https://api.eia.gov/v2/international/data/"