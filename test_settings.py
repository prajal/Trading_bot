import os
from dotenv import load_dotenv

# Force reload
load_dotenv(override=True)

print("Environment Variables:")
print(f"LIVE_TRADING_ENABLED = {os.getenv('LIVE_TRADING_ENABLED')}")
print(f"DRY_RUN_MODE = {os.getenv('DRY_RUN_MODE')}")

# Check how they're parsed
live_trading = os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true'
dry_run = os.getenv('DRY_RUN_MODE', 'true').lower() == 'true'

print(f"\nParsed Values:")
print(f"Live Trading: {live_trading}")
print(f"Dry Run: {dry_run}")
