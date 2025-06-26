import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json
import sys
import argparse
from kiteconnect import KiteConnect

API_KEY = "t4otrxd7h438r47b"
API_SECRET = "7eeyv2x2c3dje7cg3typakyzozidzbq4"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kite_tokens.json")

def save_tokens(access_token, public_token, refresh_token=None):
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump({"access_token": access_token, "public_token": public_token, "refresh_token": refresh_token}, f)

def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return None

def get_kite_session(request_token=None):
    kite = KiteConnect(api_key=API_KEY)
    tokens = load_tokens()

    if tokens and tokens.get("access_token"):
        try:
            kite.set_access_token(tokens["access_token"])
            kite.profile()
            print("Using saved access token.")
            return kite
        except Exception as e:
            print(f"Saved access token invalid: {e}")
            if tokens and tokens.get("refresh_token"):
                try:
                    print("Attempting to renew access token using refresh token...")
                    data = kite.renew_access_token(tokens["refresh_token"], API_SECRET)
                    access_token = data["access_token"]
                    public_token = data["public_token"]
                    refresh_token = data.get("refresh_token")
                    save_tokens(access_token, public_token, refresh_token)
                    kite.set_access_token(access_token)
                    print("Access token renewed successfully.")
                    return kite
                except Exception as e:
                    print(f"Failed to renew access token: {e}")

    if request_token:
        try:
            print("Generating new session using request token...")
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            access_token = data["access_token"]
            public_token = data["public_token"]
            refresh_token = data.get("refresh_token")
            save_tokens(access_token, public_token, refresh_token)
            kite.set_access_token(access_token)
            print("New session generated and tokens saved.")
            return kite
        except Exception as e:
            print(f"Error generating session: {e}")
    
    return None

def get_instrument_token(kite, symbol, exchange="NSE", instrument_type="EQ"):
    all_instruments = kite.instruments(exchange)
    for instrument in all_instruments:
        if instrument["tradingsymbol"].upper() == symbol.upper() and instrument["exchange"] == exchange and instrument["instrument_type"] == instrument_type:
            return instrument["instrument_token"], instrument
    return None, None

def list_available_instruments(kite, exchange="NSE", instrument_type="EQ", limit=20):
    all_instruments = kite.instruments(exchange)
    filtered = [i for i in all_instruments if i["instrument_type"] == instrument_type]
    print("\nAvailable NSE EQ instruments (showing up to {}):".format(limit))
    print("{:<30} | {}".format("Trading Symbol", "Name"))
    print("-"*60)
    for inst in filtered[:limit]:
        print("{:<30} | {}".format(inst['tradingsymbol'], inst['name']))
    if len(filtered) > limit:
        print(f"...and {len(filtered)-limit} more.")
    print("\nUse the Trading Symbol (left column) as the --symbol argument. Enclose in quotes if it contains spaces.")

def search_instruments(kite, query, exchange="NSE", instrument_type="EQ", limit=50):
    all_instruments = kite.instruments(exchange)
    query_lower = query.lower()
    filtered = [i for i in all_instruments if instrument_type == i["instrument_type"] and (query_lower in i["tradingsymbol"].lower() or query_lower in i["name"].lower())]
    print(f"\nSearch results for '{query}' (showing up to {limit}):")
    print("{:<30} | {}".format("Trading Symbol", "Name"))
    print("-"*60)
    for inst in filtered[:limit]:
        print("{:<30} | {}".format(inst['tradingsymbol'], inst['name']))
    if len(filtered) > limit:
        print(f"...and {len(filtered)-limit} more.")
    if not filtered:
        print("No instruments found matching your search.")
    print("\nUse the Trading Symbol (left column) as the --symbol argument. Enclose in quotes if it contains spaces.")

def download_historical_data(kite, instrument_token, symbol, start_date, end_date, interval="minute"):
    all_data = []
    current_date = start_date
    print(f'Starting data download for {symbol} from {start_date.strftime("%Y-%m-%d %H:%M:%S")} to {end_date.strftime("%Y-%m-%d %H:%M:%S")}')
    delta = timedelta(days=60)
    while current_date <= end_date:
        from_date = current_date
        to_date = min(current_date + delta, end_date)
        print(f'Fetching data for period: {from_date.strftime("%Y-%m-%d %H:%M:%S")} to {to_date.strftime("%Y-%m-%d %H:%M:%S")}')
        try:
            data = kite.historical_data(instrument_token, from_date, to_date, interval)
            if data:
                df = pd.DataFrame(data)
                all_data.append(df)
                print(f"Successfully fetched {len(df)} records.")
            else:
                print("No data received for this period.")
            time.sleep(0.5)
        except Exception as e:
            print(f'Error fetching data for {from_date.strftime("%Y-%m-%d %H:%M:%S")} to {to_date.strftime("%Y-%m-%d %H:%M:%S")}: {e}')
            time.sleep(5)
        current_date = to_date + timedelta(seconds=1)
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df["date"] = pd.to_datetime(final_df["date"])
        final_df.set_index("date", inplace=True)
        final_df.sort_index(inplace=True)
        output_file = f"{symbol.upper()}_historical_data.csv"
        final_df.to_csv(output_file)
        print(f"Successfully downloaded and saved {len(final_df)} records to {output_file}")
    else:
        print("No data was downloaded.")

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")

def main():
    parser = argparse.ArgumentParser(description="Download historical data for any NSE equity instrument.")
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., NIFTYBEES, RELIANCE, TCS, etc.)')
    parser.add_argument('--list', action='store_true', help='List available NSE EQ instruments and exit')
    parser.add_argument('--search', type=str, help='Search for trading instruments by keyword')
    parser.add_argument('--request_token', type=str, help='(Optional) Fresh request token for authentication')
    parser.add_argument('--interval', type=str, choices=['minute', 'day'], default=None, help="Data interval: 'minute' (default, last 2 years) or 'day' (default, last 10 years)")
    parser.add_argument('--years', type=int, help='Number of years of data to fetch (easier than specifying start-date). Overrides --start-date.')
    parser.add_argument('--start-date', type=parse_date, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_date, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    kite = get_kite_session(request_token=args.request_token)
    if not kite:
        print("Could not establish Kite session. Please obtain a new request_token if necessary.")
        return

    if args.search:
        search_instruments(kite, args.search)
        return

    if args.list or not args.symbol:
        if not args.symbol:
            print("\nUSAGE:")
            print("  python historical.py --symbol \"NIFTY 50\"")
            print("  python historical.py --symbol RELIANCE")
            print("  python historical.py --list  # To see more symbols")
            print("  python historical.py --search \"nifty bees etf\"  # To search")
            print("  python historical.py --symbol RELIANCE --interval day --years 10")
            print("  python historical.py --symbol RELIANCE --interval minute --years 1")
            print("  # You can still use --start-date and --end-date for custom ranges if needed")
        list_available_instruments(kite)
        if not args.symbol:
            return

    symbol = args.symbol.upper()
    instrument_token, instrument_info = get_instrument_token(kite, symbol)
    if not instrument_token:
        print(f"Instrument token for symbol '{symbol}' not found. Use --list or --search to see available instruments.")
        return
    print(f"{symbol} Instrument Token: {instrument_token} ({instrument_info['name']})")

    # Determine interval and date range
    interval = args.interval or 'minute'
    now = datetime.now()
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = now
    if args.years:
        start_date = end_date - timedelta(days=args.years*365)
    elif args.start_date:
        start_date = args.start_date
    else:
        if interval == 'minute':
            start_date = end_date - timedelta(days=1*365)
        else:
            start_date = end_date - timedelta(days=10*365)
    if interval == 'minute':
        max_days = 730  # 2 years
        if (end_date - start_date).days > max_days:
            print("\nWARNING: 'minute' interval only supports up to 2 years of data. For older data, use --interval day.")
            start_date = end_date - timedelta(days=max_days)
            print(f"Fetching only last 2 years from {start_date.date()} to {end_date.date()}.")
    print(f"\nDownloading {interval} data for {symbol} from {start_date.date()} to {end_date.date()}\n")
    download_historical_data(kite, instrument_token, symbol, start_date, end_date, interval=interval)

if __name__ == "__main__":
    main()


