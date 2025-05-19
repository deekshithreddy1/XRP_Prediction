# feature_engineer.py
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
import logging 
from config import PRICE_THRESHOLD_PERCENT, TARGET_COLUMN

logger = logging.getLogger(__name__)

def create_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators and prepares features for the model."""
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        print("Error: Input to create_features is not a valid DataFrame or is empty.")
        return pd.DataFrame()
    if 'Close' not in df_input.columns:
        print("Error: 'Close' column missing in input DataFrame.")
        return pd.DataFrame()

    df = df_input.copy() # Work on a copy

    # Simple Moving Averages
    df['SMA_7'] = SMAIndicator(close=df['Close'], window=7, fillna=False).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20, fillna=False).sma_indicator()

    # Relative Strength Index (RSI)
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14, fillna=False).rsi()

    # Moving Average Convergence Divergence (MACD)
    macd_indicator = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df['MACD_diff'] = macd_indicator.macd_diff()

    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2, fillna=False)
    df['BB_mid'] = bb_indicator.bollinger_mavg() # This is SMA_20
    df['BB_high'] = bb_indicator.bollinger_hband()
    df['BB_low'] = bb_indicator.bollinger_lband()
    df['BB_high_indicator'] = bb_indicator.bollinger_hband_indicator() # 1 if close > BB high
    df['BB_low_indicator'] = bb_indicator.bollinger_lband_indicator()   # 1 if close < BB low

    # Lagged features (price change from N days ago)
    df['Price_Change_1D'] = df['Close'].pct_change(periods=1)
    df['Price_Change_3D'] = df['Close'].pct_change(periods=3)

    # Volatility (e.g., standard deviation of returns over N periods)
    df['Volatility_7D'] = df['Close'].pct_change().rolling(window=7).std()


    # --- Create Target Variable for training ---
    df['Future_Close'] = df['Close'].shift(-1)
    df['Future_Price_Change_Ratio'] = (df['Future_Close'] - df['Close']) / df['Close']

    def assign_signal(change_ratio):
        if pd.isna(change_ratio): # For the last row where future price is unknown
            return np.nan
        if change_ratio > PRICE_THRESHOLD_PERCENT:
            return "BUY"
        elif change_ratio < -PRICE_THRESHOLD_PERCENT:
            return "SELL"
        else:
            return "HOLD"

    df[TARGET_COLUMN] = df['Future_Price_Change_Ratio'].apply(assign_signal)

    # Fill initial NaNs from rolling calculations with 0 or ffill/bfill if appropriate
    # For simplicity, we'll let the predictor handle NaNs in features before training/prediction
    # or one could fill them here, e.g.:
    # for col in ['SMA_7', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_mid', 'BB_high', 'BB_low', 'Volatility_7D', 'Price_Change_1D', 'Price_Change_3D']:
    #     if col in df.columns:
    #         df[col] = df[col].fillna(0) # Or df[col].fillna(method='bfill').fillna(method='ffill')


    print("Features created successfully.")
    return df