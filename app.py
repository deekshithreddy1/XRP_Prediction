# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime
import logging # For better logging

# Import from local modules
from config import CSV_FILE_PATH, SAVED_MODEL_PATH
from data_manager import load_or_fetch_data
from feature_engineer import create_features
from predictor import train_model, load_model, get_prediction

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Output to console
logger = logging.getLogger(__name__) # This logger can be used throughout app.py

app = Flask(__name__)

# --- Global App State (Simple Approach) ---
# These will be initialized by initialize_app_state()
current_data_with_features = pd.DataFrame()
trained_model_instance = None
prediction_details = {
    "signal": "Initializing...",
    "reason": "System starting up.",
    "last_data_date": "N/A",
    "current_price": "N/A",
    "last_model_train_time": "N/A",
    "model_status": "Not Loaded"
}

def initialize_app_state():
    """Loads data, trains/loads model, and makes initial prediction."""
    global current_data_with_features, trained_model_instance, prediction_details
    logger.info("Initializing application state...")

    raw_data = load_or_fetch_data()
    if raw_data.empty:
        logger.error("Failed to load or fetch market data.")
        prediction_details.update({
            "signal": "Error", "reason": "Could not load/fetch market data.",
            "model_status": "Data Error"
        })
        return

    current_data_with_features = create_features(raw_data.copy())
    if current_data_with_features.empty:
        logger.error("Failed to process features from market data.")
        prediction_details.update({
            "signal": "Error", "reason": "Could not process features.",
            "model_status": "Feature Error"
        })
        return

    trained_model_instance = load_model()
    if trained_model_instance:
        prediction_details["model_status"] = "Loaded from File"
        try:
            mtime = datetime.fromtimestamp(SAVED_MODEL_PATH.stat().st_mtime)
            prediction_details["last_model_train_time"] = mtime.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
             prediction_details["last_model_train_time"] = "Unknown (file exists)"
    else:
        logger.info("No pre-trained model found or failed to load. Attempting to train a new one...")
        trained_model_instance = train_model(current_data_with_features.copy()) # Pass a copy for training
        if trained_model_instance:
            prediction_details["model_status"] = "Newly Trained"
            prediction_details["last_model_train_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            logger.error("Failed to train a prediction model.")
            prediction_details.update({
                "signal": "Error", "reason": "Failed to train prediction model.",
                "model_status": "Training Failed",
                "last_model_train_time": "N/A"
            })
            return

    # Make prediction with the loaded/trained model
    signal, reason = get_prediction(current_data_with_features, trained_model_instance)
    prediction_details["signal"] = signal
    prediction_details["reason"] = reason

    if not current_data_with_features.empty:
        last_row = current_data_with_features.iloc[-1]
        prediction_details["last_data_date"] = last_row.name.strftime('%Y-%m-%d') if isinstance(last_row.name, pd.Timestamp) else str(last_row.name)
        prediction_details["current_price"] = f"${last_row['Close']:.4f}" if 'Close' in last_row else "N/A"

    logger.info(f"Initialization complete. Prediction: {signal}, Reason: {reason}")


@app.route('/')
def index():
    # Pass a copy of prediction_details to avoid modification issues if any async stuff happens
    # (though not strictly necessary in this simple sync app)
    return render_template('index.html', data=dict(prediction_details), csv_path=str(CSV_FILE_PATH.resolve()))

@app.route('/api/force_refresh', methods=['POST'])
def force_refresh_endpoint():
    logger.info("API: Forcing application state refresh...")
    initialize_app_state()
    return jsonify({"status": "success", "message": "Data and prediction refreshed.", "prediction": prediction_details})


if __name__ == '__main__':
    logger.info(f"Starting XRP Prediction App...")
    logger.info(f"Data will be stored in: {CSV_FILE_PATH.parent.resolve()}")
    logger.info(f"Model will be stored in: {SAVED_MODEL_PATH.parent.resolve()}")

    initialize_app_state() # Initial data load and model training/loading
    app.run(debug=False, host='0.0.0.0', port=5001) # Set debug=False for production-like behavior