# data_manager.py
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
import os
import logging
from config import COINGECKO_API_URL, CSV_FILE_PATH, DAYS_TO_FETCH_INITIAL, DAYS_TO_FETCH_STALE, STALE_THRESHOLD_DAYS

logger = logging.getLogger(__name__) # Use the logger from app.py or define one

def fetch_xrp_historical_data(days: int) -> pd.DataFrame:
    """Fetches historical XRP data from CoinGecko for the specified number of days."""
    if not isinstance(days, int) or days <= 0:
        logger.error("Number of days to fetch must be a positive integer.")
        return pd.DataFrame()

    logger.info(f"Attempting to fetch {days} days of data from CoinGecko...")
    # CoinGecko 'days' param: Data up to 'days' days ago from today.
    # e.g., days=1 means data for yesterday. days=0 means data for today.
    # If we want N distinct past days + today, we can ask for N days.
    # The API returns N+1 data points (from N days ago up to today).
    # If we pass `days_to_fetch -1` to API, and `days_to_fetch` is 1, API gets 0, returns 1 point (today)
    # If `days_to_fetch` is 40, API gets 39, returns 40 points.
    api_days_param = str(max(0, days - 1)) # Ensure it's not negative

    params = {
        'vs_currency': 'usd',
        'days': api_days_param,
        'interval': 'daily'
    }
    try:
        response = requests.get(COINGECKO_API_URL, params=params, timeout=20)
        response.raise_for_status()
        api_data = response.json()

        prices = api_data.get('prices', [])
        total_volumes = api_data.get('total_volumes', [])

        if not prices:
            logger.warning("No price data received from CoinGecko API.")
            return pd.DataFrame()

        # Create DataFrame from prices
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        # Convert timestamp to datetime, normalize to midnight UTC, make it the index
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.normalize()

        # Merge volumes
        if total_volumes:
            df_vol = pd.DataFrame(total_volumes, columns=['timestamp_vol', 'Volume'])
            df_vol['Date_vol'] = pd.to_datetime(df_vol['timestamp_vol'], unit='ms', utc=True).dt.normalize()
            df = pd.merge(df, df_vol[['Date_vol', 'Volume']], left_on='Date', right_on='Date_vol', how='left').drop(columns=['Date_vol'])
        else:
            df['Volume'] = 0.0 # Assign default if not available

        df['Open'] = df['Close'] # CoinGecko 'simple' price often only has close
        df['High'] = df['Close']
        df['Low'] = df['Close']

        df = df.set_index('Date')
        df = df.drop(columns=['timestamp'], errors='ignore') # Drop original timestamp
        df = df.sort_index()

        # Ensure the correct number of data points are returned.
        # If API gave N+1 points for `api_days_param = N-1`, we want N points.
        if len(df) > days and days > 0: # If days=0, API might give 1 or 2, keep all
             df = df.tail(days)
        elif len(df) == 0 and days > 0:
            logger.warning(f"CoinGecko returned 0 data points when {days} were expected.")


        logger.info(f"Successfully fetched {len(df)} data points from CoinGecko.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except requests.exceptions.Timeout:
        logger.error(f"Timeout occurred while fetching data from {COINGECKO_API_URL}")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {response.text if response else 'No response'}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during API request: {e}")
    except (ValueError, TypeError) as e: # Catches JSON parsing errors or unexpected data types
        logger.error(f"Could not parse API response or unexpected data type: {e}")
    return pd.DataFrame()


def load_or_fetch_data() -> pd.DataFrame:
    """
    Manages loading data from CSV or fetching new data based on freshness and validity.
    """
    logger.info(f"Attempting to load/fetch data. CSV path: {CSV_FILE_PATH.resolve()}")
    df_existing = pd.DataFrame()
    fetch_fresh_full = False
    days_to_fetch_new = 0 # Default to fetching no new days unless conditions are met

    current_utc_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    if CSV_FILE_PATH.exists() and CSV_FILE_PATH.stat().st_size > 0:
        try:
            # Read CSV, ensure 'Date' column is parsed as datetime and set as index
            df_existing = pd.read_csv(CSV_FILE_PATH, parse_dates=['Date'], index_col='Date')
            # Ensure index is timezone-aware (UTC) and normalized
            if df_existing.index.tz is None:
                 df_existing.index = df_existing.index.tz_localize('UTC')
            else:
                 df_existing.index = df_existing.index.tz_convert('UTC')
            df_existing.index = df_existing.index.normalize()

            logger.info(f"Successfully loaded {len(df_existing)} records from {CSV_FILE_PATH}")

            if not df_existing.empty:
                last_date_in_csv = df_existing.index.max() # Already UTC normalized
                logger.info(f"Last date in CSV: {last_date_in_csv.strftime('%Y-%m-%d')}, Current UTC date: {current_utc_date.strftime('%Y-%m-%d')}")

                if last_date_in_csv > current_utc_date:
                    logger.warning(f"Data in CSV is from the future ({last_date_in_csv.strftime('%Y-%m-%d')}). Discarding and fetching fresh initial data.")
                    df_existing = pd.DataFrame() # Invalidate current CSV data
                    fetch_fresh_full = True
                    days_to_fetch_new = DAYS_TO_FETCH_INITIAL
                else:
                    days_since_last_update = (current_utc_date - last_date_in_csv).days
                    logger.info(f"Days since last CSV update: {days_since_last_update}")

                    if days_since_last_update == 0: # Data is current up to today
                        logger.info("Data in CSV is current as of today. No new data to fetch.")
                        days_to_fetch_new = 0
                    elif days_since_last_update > STALE_THRESHOLD_DAYS:
                        logger.info(f"Data is older than {STALE_THRESHOLD_DAYS} days. Fetching fresh {DAYS_TO_FETCH_STALE} days.")
                        fetch_fresh_full = True
                        days_to_fetch_new = DAYS_TO_FETCH_STALE
                    elif days_since_last_update > 0:
                        days_to_fetch_new = days_since_last_update
                        logger.info(f"Fetching data for the last {days_to_fetch_new} missing day(s).")
                    # days_since_last_update < 0 is handled by the future date check
            else: # CSV exists but was parsed as empty (e.g., only headers)
                logger.info(f"{CSV_FILE_PATH} exists but is empty. Fetching initial data.")
                fetch_fresh_full = True
                days_to_fetch_new = DAYS_TO_FETCH_INITIAL
        except pd.errors.EmptyDataError:
            logger.warning(f"{CSV_FILE_PATH} is empty. Fetching initial data.")
            fetch_fresh_full = True
            days_to_fetch_new = DAYS_TO_FETCH_INITIAL
        except Exception as e:
            logger.error(f"Error reading or processing {CSV_FILE_PATH}: {e}. Will attempt to fetch fresh initial data.")
            df_existing = pd.DataFrame() # Invalidate on error
            fetch_fresh_full = True
            days_to_fetch_new = DAYS_TO_FETCH_INITIAL
    else: # CSV does not exist or is 0 bytes
        logger.info(f"{CSV_FILE_PATH} not found or is 0 bytes. Fetching initial data.")
        fetch_fresh_full = True
        days_to_fetch_new = DAYS_TO_FETCH_INITIAL

    # Proceed to fetch new data if needed
    if days_to_fetch_new > 0:
        logger.info(f"Proceeding to fetch {days_to_fetch_new} day(s) of new data. Full refresh: {fetch_fresh_full}")
        df_new = fetch_xrp_historical_data(days=days_to_fetch_new)
        time.sleep(1) # API courtesy delay

        if not df_new.empty:
            if fetch_fresh_full or df_existing.empty:
                df_to_save = df_new
                logger.info("New full dataset prepared for saving.")
            else:
                # Append new data, ensuring no duplicates if there's an overlap
                # Only add rows from df_new that are not already in df_existing's index
                df_to_save = pd.concat([df_existing, df_new[~df_new.index.isin(df_existing.index)]])
                logger.info(f"Appended {len(df_new[~df_new.index.isin(df_existing.index)])} new rows.")

            df_to_save = df_to_save.sort_index()
            # Remove exact duplicate rows just in case (should be rare with index logic)
            df_to_save = df_to_save[~df_to_save.index.duplicated(keep='last')]

            try:
                df_to_save.to_csv(CSV_FILE_PATH)
                logger.info(f"Data successfully saved to {CSV_FILE_PATH} ({len(df_to_save)} rows).")
                return df_to_save
            except IOError as e:
                logger.error(f"IOError saving data to {CSV_FILE_PATH}: {e}")
                return df_existing # Fallback to previously loaded data if save fails
            except Exception as e:
                logger.error(f"Unexpected error saving data to {CSV_FILE_PATH}: {e}")
                return df_existing
        else:
            logger.warning("No new data was fetched. Returning previously loaded data if available.")
            return df_existing # Return existing if new fetch failed
    elif not df_existing.empty: # No new days to fetch, and existing data is loaded and valid
        logger.info("Returning existing, up-to-date data.")
        return df_existing
    else: # No new days to fetch, and no valid existing data (e.g. initial run failed, or CSV was invalid and cleared)
        logger.error("No data could be loaded or fetched. Returning empty DataFrame.")
        return pd.DataFrame()