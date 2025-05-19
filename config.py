# config.py
import os
from pathlib import Path

# --- API Configuration ---
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/ripple/market_chart"

# --- Data Storage Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "trading_data" / "XRP"
MODEL_STORAGE_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE_PATH = DATA_DIRECTORY / "xrp_daily_data.csv"
SAVED_MODEL_PATH = MODEL_STORAGE_DIR / "xrp_predictor_model.joblib"

# --- Data Fetching Logic ---
DAYS_TO_FETCH_INITIAL = 300
DAYS_TO_FETCH_STALE = 300
STALE_THRESHOLD_DAYS = 30

# --- Model & Prediction ---
TARGET_COLUMN = 'Signal'
PRICE_THRESHOLD_PERCENT = 0.015 # 1.5% change threshold for BUY/SELL signal labeling

# Features to be used by the model
FEATURES = [
    'SMA_7', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
    'BB_high_indicator', 'BB_low_indicator', # Bollinger Band indicators (binary)
    'Price_Change_1D', 'Price_Change_3D', 'Volatility_7D'
]