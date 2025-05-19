# predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving/loading model
from pathlib import Path
from config import TARGET_COLUMN

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xrp_predictor_model.joblib"

FEATURES = [
    'SMA_7', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
    'BB_high', 'BB_low', 'Price_Change_1D', 'Price_Change_3D',
    # Add more features if desired, e.g., relative position to BB bands
    # 'Close' could also be a feature if scaled properly, or differences from MAs
]

def train_model(df_featured):
    """Trains a RandomForestClassifier model and saves it."""
    df_train = df_featured.copy()

    # Ensure TARGET_COLUMN is present and drop rows where it's NaN (last row)
    if TARGET_COLUMN not in df_train.columns:
        print(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
        return None
    df_train = df_train.dropna(subset=[TARGET_COLUMN]) # Remove rows where target is NaN

    # Also drop rows where any feature is NaN
    df_train = df_train.dropna(subset=FEATURES)

    if df_train.empty or len(df_train) < 20: # Need enough data to train
        print("Not enough data to train the model after NaN removal.")
        return None

    X = df_train[FEATURES]
    y = df_train[TARGET_COLUMN]

    if len(y.unique()) < 2:
        print(f"Not enough classes in target variable to train. Found: {y.unique()}")
        return None

    # Time series split: train on older data, test on newer.
    # For this simple on-the-fly training, we'll use a simpler split.
    # A more robust approach would use TimeSeriesSplit from sklearn.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False # IMPORTANT: shuffle=False for time series
    )

    if len(X_train) < 1 or len(X_test) < 1:
        print("Train or test set is empty after split. Cannot train.")
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during model fitting: {e}")
        print(f"Unique values in y_train: {y_train.unique()}")
        return None


    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\n--- Model Training Evaluation (on test split) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    try:
        print("Classification Report (Test Set):\n", classification_report(y_test, y_pred_test, zero_division=0))
    except ValueError:
        print("Could not generate classification report (likely due to missing classes in y_test or y_pred).")


    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return model

def load_model():
    """Loads the trained model."""
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Will retrain if possible.")
            return None
    return None

def get_prediction(df_featured, model):
    """
    Makes a prediction for the latest available data point.
    The prediction is for the *next* period (e.g., tomorrow).
    """
    if model is None:
        print("Model not available for prediction.")
        return "N/A", "Model not trained."
    if df_featured.empty:
        print("No data to make a prediction on.")
        return "N/A", "No data."

    latest_data_point = df_featured.iloc[-1:] # Get the last row as a DataFrame
    latest_features = latest_data_point[FEATURES]

    if latest_features.isnull().any().any():
        missing_cols = latest_features.columns[latest_features.isnull().any()].tolist()
        print(f"Warning: Latest data point has NaN values in features: {missing_cols}. Prediction might be unreliable.")
        # Option: fillna with a strategy (e.g., ffill from df_featured, or mean)
        # For now, we'll proceed, but a robust system would handle this.
        # latest_features = latest_features.fillna(method='ffill').fillna(0) # Example fill

    if latest_features.dropna().empty: # If all features are NaN after potential fill
        return "N/A", "Features for prediction are all NaN."

    try:
        prediction = model.predict(latest_features)[0]
        # Optional: Add prediction probability for more nuance
        # probabilities = model.predict_proba(latest_features)[0]
        # prob_dict = dict(zip(model.classes_, probabilities))
        # reason = f"Predicted based on current market indicators. Confidence: {prob_dict.get(prediction, 0):.2f}"
        reason = "Predicted based on historical patterns and current technical indicators."
        return prediction, reason
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", str(e)

if __name__ == '__main__':
    from data_manager import load_or_fetch_data
    from feature_engineer import create_features

    print("Testing predictor standalone...")
    raw_data = load_or_fetch_data()
    if not raw_data.empty:
        featured_data = create_features(raw_data.copy())
        if not featured_data.empty:
            # Try loading model, if not found, train one
            
            model = load_model()
            if model is None:
                print("No pre-trained model found, training a new one...")
                model = train_model(featured_data)

            if model:
                signal, reason = get_prediction(featured_data, model)
                print(f"\n--- Prediction for Next Period ---")
                print(f"Date of Last Data: {featured_data.index[-1].strftime('%Y-%m-%d')}")
                print(f"Predicted Signal: {signal}")
                print(f"Reason: {reason}")
            else:
                print("Could not train or load a model.")
        else:
            print("Could not create features.")
    else:
        print("Could not load or fetch data.")