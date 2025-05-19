import os
import json
import random
from datetime import datetime, timedelta

import requests
from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

DATA_FILE = "data_store.json"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=ripple&vs_currencies=usd"

# This is the "memory" of our simulation's current price.
# It's initialized and then updated by real prices or simulation steps.
SIMULATED_CURRENT_PRICE = 0.50  # Default starting point

# --- Helper Functions ---
def load_predictions():
    """Loads predictions from the JSON file, with fallback for corruption or non-existence."""
    default_data = {
        "current_price_usd": None,
        "last_updated": None,
        "today_prediction": {
            "action": "INITIALIZING",
            "reason": "System starting up. First prediction soon!",
            "emoji": "🚀"
        },
        "next_7_days": {
            "trend": "Aligning with cosmic financial waves...",
            "target_price": "Calculating...",
            "emoji": "🌌"
        },
        "next_30_days_outlook": "The grand 30-day plan is unfolding...",
        "next_30_days_daily": [],
        "next_year": {
            "outlook": "The annual prophecy is being inscribed.",
            "potential_price": "To be revealed.",
            "emoji": "📜"
        }
    }
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{DATA_FILE}' is corrupted or unreadable. Using default data and attempting to overwrite.")
            # Optionally, backup corrupted file: os.rename(DATA_FILE, DATA_FILE + ".corrupted")
            return default_data
    return default_data


def save_predictions(data):
    """Saves the predictions data to the JSON file."""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving predictions to {DATA_FILE}: {e}")


def fetch_current_xrp_price():
    """Fetches the current XRP price from CoinGecko."""
    try:
        response = requests.get(COINGECKO_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = data.get("ripple", {}).get("usd")
        if price is not None:
            return float(price)
        print("Warning: 'usd' price not found in CoinGecko response for ripple.")
        return None
    except requests.exceptions.Timeout:
        print(f"Error fetching XRP price: Timeout from {COINGECKO_API_URL}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching XRP price: {e}")
        return None
    except (ValueError, TypeError) as e: # Catch issues if price is not a float
        print(f"Error processing price data from CoinGecko: {e}")
        return None

# --- "Magical" Prediction Logic ---

def generate_30_day_detailed_prediction(predictions_data, starting_price_for_plan):
    """
    Generates a "dynamic" 30-day prediction list.
    The plan starts from 'starting_price_for_plan'.
    """
    global SIMULATED_CURRENT_PRICE # This function might influence the base for the next overall prediction cycle if called directly.
                                   # However, its primary role is to populate predictions_data.

    print(f"Generating new 30-day plan starting from simulated price: {starting_price_for_plan:.4f}")
    daily_predictions = []
    # Use a local variable for the price progression within this 30-day plan
    price_for_30_day_plan_calc = starting_price_for_plan

    for day_num in range(1, 31):
        day_movement_roll = random.random()
        bias = 0.0 # Neutral bias by default

        if day_num <= 10: bias = 0.2
        elif day_num <= 20: bias = -0.1
        else: bias = 0.15

        effective_roll = day_movement_roll + bias

        if effective_roll > 0.6: # Trend UP (increased threshold for more variability)
            change_percent = random.uniform(0.01, 0.045) # 1% to 4.5% UP
            movement = "Anticipate UPWARD movement"
            emoji = "🔼"
            tip = "Good day to add to your XRP treasure chest!"
            action_raw = "BUY"
        elif effective_roll < 0.20: # Trend DOWN (decreased threshold for more variability)
            change_percent = random.uniform(-0.03, -0.005) # 0.5% to 3% DOWN
            movement = "Expect a slight DIP"
            emoji = "🔽"
            tip = "A small dip might be a chance, or just watch!"
            action_raw = "SELL"
        else: # Sideways / HOLD
            change_percent = random.uniform(-0.005, 0.005)
            movement = "Likely SIDEWAYS consolidation"
            emoji = "↔️"
            tip = "Patience, young trader! XRP is gathering power."
            action_raw = "HOLD"

        price_for_30_day_plan_calc *= (1 + change_percent)
        price_for_30_day_plan_calc = max(0.01, price_for_30_day_plan_calc) # Floor price

        daily_predictions.append({
            "day": day_num,
            "movement": movement,
            "emoji": emoji,
            "tip": tip,
            "action_raw": action_raw,
            "target_sim_price": price_for_30_day_plan_calc
        })

    predictions_data["next_30_days_daily"] = daily_predictions
    final_price_of_plan = price_for_30_day_plan_calc
    predictions_data["next_30_days_outlook"] = (
        f"My crystal ball shows a fascinating 30-day journey for XRP, starting from around ${starting_price_for_plan:.4f} "
        f"and potentially reaching around ${final_price_of_plan:.4f} by the end of this period! "
        "Follow the daily guide below. Each day is a step to victory!"
    )
    print(f"30-day plan generated, ending at simulated price: {final_price_of_plan:.4f}")
    # The global SIMULATED_CURRENT_PRICE is updated in generate_all_predictions based on *today's* outcome.
    # This function primarily populates the plan.


def generate_all_predictions():
    """
    The core prediction generation. Called daily.
    This function uses and MODIFIES the global SIMULATED_CURRENT_PRICE.
    """
    global SIMULATED_CURRENT_PRICE # Declare global at the very top of the function

    print(f"[{datetime.now()}] Generating new predictions...")
    predictions_data = load_predictions()

    # 1. Get current REAL price
    current_real_price_fetched = fetch_current_xrp_price()

    if current_real_price_fetched is not None:
        predictions_data["current_price_usd"] = current_real_price_fetched
        # Re-anchor SIMULATED_CURRENT_PRICE to reality if it's the very first fetch
        # or if it has drifted significantly. This keeps the simulation somewhat grounded.
        if SIMULATED_CURRENT_PRICE == 0.50 or \
           abs(SIMULATED_CURRENT_PRICE - current_real_price_fetched) / current_real_price_fetched > 0.15: # Drifted by >15%
            print(f"Re-anchoring SIMULATED_CURRENT_PRICE from {SIMULATED_CURRENT_PRICE:.4f} to real price {current_real_price_fetched:.4f}")
            SIMULATED_CURRENT_PRICE = current_real_price_fetched
    else:
        # If API fails, use the last known real price from storage if available.
        # If not, SIMULATED_CURRENT_PRICE will continue from its last simulated value.
        if predictions_data["current_price_usd"] is not None:
             print(f"API fetch failed. Using last known real price: {predictions_data['current_price_usd']:.4f}")
        else:
             print(f"API fetch failed and no last known real price. Continuing simulation from: {SIMULATED_CURRENT_PRICE:.4f}")


    predictions_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # This will be the simulated price *after* today's predicted action.
    # It starts as the current SIMULATED_CURRENT_PRICE (which is end of *last* day / start of *this* day)
    effective_simulated_price_for_today = SIMULATED_CURRENT_PRICE

    # 2. Today's Prediction (from 30-day plan or fallback)
    action_today = "HOLD" # Default
    reason_today = "Price is consolidating."
    emoji_today = "🧘"

    if predictions_data.get("next_30_days_daily") and len(predictions_data["next_30_days_daily"]) > 0:
        todays_forecast_from_plan = predictions_data["next_30_days_daily"].pop(0) # Consume one day
        action_today = todays_forecast_from_plan["action_raw"]
        effective_simulated_price_for_today = todays_forecast_from_plan['target_sim_price'] # This is the key update

        if "UP" in todays_forecast_from_plan["movement"]:
            reason_today = f"As foreseen in the 30-day plan, XRP aims for {effective_simulated_price_for_today:.4f}! Accumulate wisdom (and XRP!)."
            emoji_today = "📈"
        elif "DOWN" in todays_forecast_from_plan["movement"]:
            reason_today = f"A planned dip towards {effective_simulated_price_for_today:.4f}. Smart planners see opportunities."
            emoji_today = "📉"
        else: # HOLD or Sideways
            reason_today = f"Consolidating around {effective_simulated_price_for_today:.4f}, as per the grand design. Patience."
            emoji_today = "🧘"
    else: # Fallback if 30-day plan is exhausted (should be rare due to auto-regen)
        print("Warning: 30-day plan exhausted or missing. Generating simple daily prediction.")
        action_roll = random.random()
        if action_roll < 0.6:
            action_today = "BUY"
            reason_today = "Positive momentum detected! A good day for growth."
            emoji_today = "🚀"
            effective_simulated_price_for_today *= random.uniform(1.005, 1.025)
        elif action_roll < 0.85:
            action_today = "HOLD"
            # reason_today, emoji_today already set to HOLD defaults
            effective_simulated_price_for_today *= random.uniform(0.998, 1.002)
        else:
            action_today = "CAUTION"
            reason_today = "A minor correction predicted. Observe or prepare."
            emoji_today = "🧐"
            effective_simulated_price_for_today *= random.uniform(0.985, 0.995)
        effective_simulated_price_for_today = max(0.01, effective_simulated_price_for_today)


    predictions_data["today_prediction"] = {
        "action": action_today,
        "reason": reason_today,
        "emoji": emoji_today
    }

    # CRITICAL: Update the global SIMULATED_CURRENT_PRICE to reflect the outcome of *today's* prediction.
    # This new value will be the starting point for the *next* day's simulation or if the 30-day plan is regenerated.
    SIMULATED_CURRENT_PRICE = effective_simulated_price_for_today

    # 3. Regenerate 30-day plan if it's running low.
    # It will start from the NEWLY updated SIMULATED_CURRENT_PRICE.
    if not predictions_data.get("next_30_days_daily") or len(predictions_data["next_30_days_daily"]) < 25:
        print("Regenerating 30-day detailed prediction plan...")
        generate_30_day_detailed_prediction(predictions_data, SIMULATED_CURRENT_PRICE)


    # 4. Next 7 Days Outlook (derived from the updated 30-day plan)
    if predictions_data["next_30_days_daily"] and len(predictions_data["next_30_days_daily"]) >= 7:
        seven_day_slice = predictions_data["next_30_days_daily"][:7]
        num_up = sum(1 for day in seven_day_slice if "UP" in day["movement"])
        num_down = sum(1 for day in seven_day_slice if "DOWN" in day["movement"])
        final_price_7_days_sim = seven_day_slice[-1]["target_sim_price"]
        trend_7d, emoji_7d = "Calculating...", "🤔"

        if num_up > num_down + 2:
            trend_7d, emoji_7d = "Strong upward trajectory expected!", "⏫"
        elif num_down > num_up + 1:
            trend_7d, emoji_7d = "A corrective phase, opportunities arise.", "⏬"
        else:
            trend_7d, emoji_7d = "Mixed movements, preparing for growth.", "📊"

        predictions_data["next_7_days"] = {
            "trend": trend_7d,
            "target_price": f"Around ${final_price_7_days_sim:.4f} (simulated)",
            "emoji": emoji_7d
        }
    else: # Fallback if 30-day plan too short (should be rare)
        predictions_data["next_7_days"] = {
            "trend": "Generally positive with fluctuations. Big moves brewing!",
            "target_price": f"Aiming for ~${SIMULATED_CURRENT_PRICE * 1.07:.4f} (simulated, +7%)",
            "emoji": "📈"
        }

    # 5. Next Year Prediction (based on current SIMULATED_CURRENT_PRICE for fun)
    # Use current_real_price_fetched if available for a more "grounded" spectacular prediction
    base_for_yearly_pred = current_real_price_fetched if current_real_price_fetched is not None else SIMULATED_CURRENT_PRICE
    
    # Ensure base_for_yearly_pred is not None before multiplication
    if base_for_yearly_pred is None:
        base_for_yearly_pred = 0.50 # Ultimate fallback if everything else fails
        print("Warning: Could not determine a base price for yearly prediction, using default.")

    year_multiplier = random.uniform(2.5, 8.0) # Adjusted for a bit more realism in "prediction"
    year_target_price = base_for_yearly_pred * year_multiplier

    possible_events = [
        "major global adoption announcements", "breakthrough tech integration",
        "positive regulatory clarity", "XRP becoming a key bridge currency"
    ]
    predictions_data["next_year"] = {
        "outlook": f"A MONUMENTAL year ahead for XRP! We foresee {random.choice(possible_events)} driving its value. For the 100x thinkers!",
        "potential_price": f"Could reach ${year_target_price:.2f} or higher! (Based on today's outlook)",
        "emoji": "🌠"
    }

    save_predictions(predictions_data)
    print(f"[{datetime.now()}] Predictions updated. Current simulated price: {SIMULATED_CURRENT_PRICE:.4f}")


# --- Flask Routes ---
@app.route('/')
def index():
    predictions = load_predictions()
    # Ensure current_price_usd is float for formatting, or provide default
    if predictions.get("current_price_usd") is not None:
        try:
            predictions["current_price_usd"] = float(predictions["current_price_usd"])
        except (ValueError, TypeError):
            predictions["current_price_usd"] = 0.0 # Fallback for display
    else:
        # If current_price_usd is None, provide a value so Jinja formatting doesn't break
        predictions["current_price_usd"] = 0.0 # Indicates data not available yet

    return render_template('index.html', data=predictions)

@app.route('/api/predictions')
def api_predictions():
    return jsonify(load_predictions())

# --- Scheduler Setup ---
scheduler = BackgroundScheduler(daemon=True, timezone="UTC") # Explicitly set timezone
# Schedule job to run daily at 10:00 AM UTC
scheduler.add_job(generate_all_predictions, 'cron', hour=10, minute=0)
# For testing: scheduler.add_job(generate_all_predictions, 'interval', seconds=30)

if __name__ == '__main__':
    # Initialize global SIMULATED_CURRENT_PRICE on startup
    # Try to use a real price first, then stored simulation, then default.
    print("Application starting up...")
    temp_initial_data = load_predictions() # Load once to check stored values

    # Attempt to set SIMULATED_CURRENT_PRICE from a real fetch first
    initial_real_price = fetch_current_xrp_price()
    if initial_real_price is not None:
        SIMULATED_CURRENT_PRICE = initial_real_price
        print(f"Initialized SIMULATED_CURRENT_PRICE from live CoinGecko data: {SIMULATED_CURRENT_PRICE:.4f}")
        if temp_initial_data.get("current_price_usd") is None: # Also update stored real price if it was missing
            temp_initial_data["current_price_usd"] = initial_real_price
            # No need to save here, generate_all_predictions will handle it
    elif temp_initial_data.get("today_prediction", {}).get("action") != "INITIALIZING":
        # If API failed, but we have past simulation data, try to continue from there.
        # A simple way: check if 'next_30_days_daily' exists and has prices.
        # For now, we will let generate_all_predictions handle SIMULATED_CURRENT_PRICE re-anchoring or continuation.
        # The global SIMULATED_CURRENT_PRICE (0.50 default) will be used if API fails and no better re-anchor point.
        print(f"Initial API fetch failed. Will rely on generate_all_predictions logic for SIMULATED_CURRENT_PRICE.")


    # Run initial prediction generation on startup.
    # This will:
    # 1. Fetch real price (again, but it's fine, or use 'initial_real_price' if desired to optimize).
    # 2. Update/Re-anchor SIMULATED_CURRENT_PRICE.
    # 3. Populate/Regenerate 30-day plan using the (potentially new) SIMULATED_CURRENT_PRICE.
    # 4. Generate today's, weekly, yearly predictions.
    # 5. Save everything.
    print("Running initial prediction generation on startup...")
    generate_all_predictions()

    scheduler.start()
    print("Scheduler started. Predictions will update daily at 10 AM UTC.") # Clarified UTC
    print("Access the app at http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0') # Use 0.0.0.0 to make it accessible on your network if needed