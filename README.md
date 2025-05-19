<div align="center">
  <h1>📈 XRP Price Movement Predictor 🚀</h1>
  <p>
    A Flask-based web application that leverages machine learning to predict potential short-term price movements (BUY, SELL, HOLD) for XRP.
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
    <img src="https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask" alt="Flask Version">
    <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/Pandas-Data_Manipulation-blueviolet?style=for-the-badge&logo=pandas" alt="Pandas">
  </p>
</div>

---

## 🌟 Overview

This project provides a simple yet effective framework for predicting XRP price trends. It fetches historical XRP data, engineers relevant technical features, trains a Random Forest model, and presents the latest prediction (BUY, SELL, or HOLD) through a clean web interface.

The primary goal is to offer an illustrative example of how one might approach time-series forecasting for cryptocurrencies using common data science tools. <span style="color:orange;">**It is intended for educational and illustrative purposes and should NOT be considered financial advice.**</span>

---

## ✨ Features

*   **📊 Data Management**: Automatically fetches historical XRP data (or loads from a local CSV).
*   **🛠️ Feature Engineering**: Calculates various technical indicators:
    *   Simple Moving Averages (SMA 7, SMA 20)
    *   Relative Strength Index (RSI 14)
    *   Moving Average Convergence Divergence (MACD)
    *   Bollinger Bands
    *   Price Change (1D, 3D)
    *   Volatility (7D)
*   **🤖 Machine Learning Model**:
    *   Trains a `RandomForestClassifier` on the engineered features.
    *   Saves the trained model for quick reloading.
    *   Retrains if no model is found or if a refresh is triggered.
*   **🖥️ Web Interface**:
    *   Displays the current prediction (BUY/SELL/HOLD) and the reasoning.
    *   Shows key details like last data date, current price, and model status.
    *   Built with Flask and a simple HTML template.
*   **🔄 Force Refresh**: API endpoint to manually trigger a data refresh and model retrain.
*   **📝 Logging**: Comprehensive logging for easier debugging and monitoring.

---

---

## ⚙️ How It Works - The Prediction Pipeline

1.  **<span style="color:dodgerblue;">Data Acquisition (`data_manager.py`)</span>**:
    *   The application first attempts to load historical XRP price data from a local CSV file (specified in `config.py`, e.g., `trading_data/xrp_data.csv`).
    *   If the CSV doesn't exist or is outdated, it (ideally, if implemented in `data_manager.py`) fetches fresh data from a crypto API (e.g., CoinGecko, Binance). *You'll need to ensure `data_manager.py` handles this and `config.py` has any necessary API keys.*

2.  **<span style="color:mediumseagreen;">Feature Engineering (`feature_engineer.py`)</span>**:
    *   The raw price data (Open, High, Low, Close, Volume) is processed.
    *   A suite of technical indicators (SMAs, RSI, MACD, Bollinger Bands, etc.) and lagged features are calculated.
    *   A target variable (`Signal_Target`: BUY, SELL, HOLD) is created based on future price changes exceeding a predefined `PRICE_THRESHOLD_PERCENT` (from `config.py`).

3.  **<span style="color:coral;">Model Training/Loading (`predictor.py`)</span>**:
    *   The system checks if a pre-trained model (`models/xrp_predictor_model.joblib`) exists.
    *   **If Yes**: The model is loaded.
    *   **If No (or load fails)**: A new `RandomForestClassifier` model is trained using the engineered features and the target variable. The data is split chronologically (older data for training, newer for testing) to simulate real-world prediction. The trained model is then saved.

4.  **<span style="color:orchid;">Prediction (`predictor.py` & `app.py`)</span>**:
    *   The latest available data point with its engineered features is fed into the loaded/trained model.
    *   The model predicts the next period's signal (BUY, SELL, or HOLD).

5.  **<span style="color:gold;">Display (`app.py` & `templates/index.html`)</span>**:
    *   The Flask application serves an HTML page.
    *   This page displays the prediction, a brief reason, the last data point's date and price, and the model's status (loaded/newly trained).

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.7+
*   pip (Python package installer)
*   Git

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/deekshithreddy1/XRP_Prediction.git
    cd XRP_Prediction
    ```

2.  **Create and Activate a Virtual Environment:**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *This will install Flask, Pandas, Scikit-learn, TA-Lib (or `ta`), joblib, etc.*

4.  **Configuration (`config.py`):**
    Create a `config.py` file in the root of the `XRP` directory. This file will hold your settings.
    Based on the imports in `app.py`, it likely contains:
    ```python
    # config.py
    from pathlib import Path

    # --- Data Configuration ---
    # If you are fetching data from an API, add your API_KEY here
    # API_KEY = "YOUR_CRYPTO_API_KEY" # Example, adjust as per your data_manager.py
    # API_SECRET = "YOUR_CRYPTO_API_SECRET" # Example

    # Path to the CSV file where historical data is stored/will be saved
    CSV_FILE_PATH = Path("trading_data") / "xrp_data.csv"

    # --- Model Configuration ---
    # Path to save/load the trained model
    SAVED_MODEL_PATH = Path("models") / "xrp_predictor_model.joblib"

    # --- Feature Engineering & Target Variable ---
    # Percentage change threshold to define a BUY or SELL signal for training
    PRICE_THRESHOLD_PERCENT = 0.01  # e.g., 1% change up for BUY, 1% change down for SELL
    TARGET_COLUMN = "Signal_Target" # Name of the target column for the model

    # Ensure parent directories exist
    CSV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ```
    <span style="color:yellow;">**Important:**</span> You will need to populate `trading_data/xrp_data.csv` with initial historical data for XRP if your `data_manager.py` doesn't automatically fetch it on the first run. The CSV should at least contain columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

5.  **Initial Data (Crucial!)**:
    *   If your `data_manager.py` is set up to fetch data when `xrp_data.csv` is missing, it might do so on the first run.
    *   Otherwise, you **must** provide an initial `xrp_data.csv` file in the `trading_data` folder. It should have columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`. The `Date` column should be parsable by Pandas.

### Running the Application

1.  Ensure your virtual environment is activated.
2.  Navigate to the `XRP` directory in your terminal.
3.  Run the Flask application:
    ```bash
    python app.py
    ```
4.  Open your web browser and go to: `http://localhost:5001` (or `http://0.0.0.0:5001`)

---

## 🎈 Using the Application

Once the application is running:

1.  **<span style="color:cyan;">Main Page (`/`)</span>**:
    *   The homepage will display the latest prediction:
        *   <span style="color:lightgreen;">**Signal**</span>: BUY, SELL, or HOLD.
        *   <span style="color:lightcoral;">**Reason**</span>: A brief explanation for the prediction.
        *   **Last Data Date**: The date of the most recent data point used.
        *   **Current Price**: The closing price from the last data point.
        *   **Model Status**: Indicates if the model was loaded from a file or newly trained.
        *   **Last Model Train Time**: Timestamp of when the model was last trained/loaded.
        *   **CSV Path**: The resolved path to the data file being used.

2.  **<span style="color:mediumpurple;">Force Refresh (API - `/api/force_refresh`)</span>**:
    *   To manually trigger a data refresh (if `data_manager.py` supports fetching), feature re-engineering, and model retraining, you can send a `POST` request to the `/api/force_refresh` endpoint.
    *   You can do this using tools like `curl` or Postman:
        ```bash
        curl -X POST http://localhost:5001/api/force_refresh
        ```
    *   This is useful if you've updated the underlying data or want to ensure the model is using the absolute latest information. The webpage will automatically reflect the new prediction after a refresh. *(Note: The current `index.html` might not automatically update without a page reload unless JavaScript is added to poll this endpoint or use its response).*

---

## 🛠️ Key Files Explained

*   **`app.py`**: The heart of the web application. It initializes Flask, defines routes (web pages and API endpoints), orchestrates the data loading, feature engineering, model training/prediction, and renders the HTML template.
*   **`config.py`**: Centralized configuration for file paths, API keys (if any), model parameters (like `PRICE_THRESHOLD_PERCENT`), and other settings. **This is the first file you should check and customize.**
*   **`data_manager.py`**: Responsible for loading data from the CSV file. Ideally, it would also handle fetching new data from an external API if the local data is missing or outdated.
*   **`feature_engineer.py`**: Takes the raw price data and computes various technical indicators (SMA, RSI, MACD, Bollinger Bands) that serve as input features for the machine learning model. It also defines the target variable for training.
*   **`predictor.py`**: Manages the machine learning model. It includes functions to train a `RandomForestClassifier`, save the trained model to disk, load an existing model, and make predictions on new data.
*   **`requirements.txt`**: Lists all Python libraries required for the project. `pip install -r requirements.txt` installs them.
*   **`templates/index.html`**: The HTML structure for the web page displayed to the user. It uses Flask's templating engine to show dynamic data.
*   **`static/style.css`**: Contains CSS rules for styling the `index.html` page.

---

## 💡 Potential Enhancements & Customization

*   **Advanced `data_manager.py`**: Implement robust data fetching from a reliable crypto API (e.g., Binance, CoinGecko, Alpha Vantage) with error handling and rate limiting.
*   **More Sophisticated Models**: Experiment with other models like Gradient Boosting, LSTM (for time series), or Prophet.
*   **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to find optimal parameters for your chosen model.
*   **Cross-Validation**: Implement time-series cross-validation (e.g., `TimeSeriesSplit` from scikit-learn) for more robust model evaluation.
*   **Real-time Updates**: Use WebSockets or AJAX polling on the frontend to update predictions without page reloads.
*   **Backtesting Framework**: Develop a system to test the prediction strategy on historical data to evaluate its hypothetical performance.
*   **User Authentication**: If deploying, add user accounts.
*   **Task Scheduling**: Use a scheduler (like APScheduler or Celery with Redis/RabbitMQ) to automatically update data and retrain the model periodically.
*   **Dockerize**: Package the application in a Docker container for easier deployment.

---

## ⚠️ Disclaimer

This project is for <span style="color:orange;">**educational and demonstrative purposes only**</span>. Cryptocurrency markets are highly volatile and speculative. Predictions made by this model are not guaranteed to be accurate and **should not be used as financial advice or the basis for any investment decisions.** Always do your own research (DYOR) and consult with a qualified financial advisor before making any investments. Past performance is not indicative of future results.

---

<div align="center">
  Happy Coding & May Your Predictions Be Ever in Your Favor (Statistically Speaking)! 🎉
</div>
