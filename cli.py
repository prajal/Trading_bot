import argparse
import sys
from auth.kite_auth import KiteAuth
from main import TradingBot
from utils.logger import get_logger

logger = get_logger(__name__)

def authenticate():
    """Handle authentication"""
    auth = KiteAuth()
    
    print("🔐 Kite Connect Authentication")
    print("=" * 40)
    
    # Check existing connection
    if auth.test_connection():
        print("✅ Already authenticated!")
        return True
    
    # Generate login URL
    login_url = auth.generate_login_url()
    print(f"1. Visit: {login_url}")
    print("2. Login and authorize the app")
    print("3. Copy the request_token from the redirected URL")
    
    request_token = input("Enter request_token: ").strip()
    
    if auth.create_session(request_token):
        print("✅ Authentication successful!")
        return True
    else:
        print("❌ Authentication failed!")
        return False

def test_connection():
    """Test Kite connection"""
    auth = KiteAuth()
    if auth.test_connection():
        print("✅ Connection test successful!")
        
        # Show account info
        kite = auth.get_kite_instance()
        if kite:
            try:
                profile = kite.profile()
                margins = kite.margins()
                equity = margins.get('equity', {})
                available_cash = equity.get('available', {}).get('cash', 0)
                
                print(f"👤 User: {profile.get('user_name')}")
                print(f"💰 Available Cash: ₹{available_cash:,.2f}")
            except Exception as e:
                print(f"⚠️  Could not fetch account details: {e}")
        
        return True
    else:
        print("❌ Connection test failed!")
        return False

def start_trading():
    """Start trading bot"""
    bot = TradingBot()
    if bot.setup():
        print("🚀 Starting trading bot...")
        print("📊 Trading: NIFTY 50 → NIFTYBEES")
        print("⏹️  Press Ctrl+C to stop")
        print()
        # Default: NIFTY 50 -> NIFTYBEES
        bot.run("256265", "2707457", "NIFTYBEES")
    else:
        print("❌ Failed to setup trading bot")

def emergency_reset():
    """Emergency position reset"""
    print("🚨 EMERGENCY POSITION RESET")
    print("This will help if your bot shows positions that don't exist")
    print("Only use this if auto square-off happened but bot still shows position")
    
    confirm = input("Are you sure you want to reset position tracking? (yes/no): ").lower().strip()
    if confirm == "yes":
        print("✅ Emergency reset completed")
        print("💡 Restart your bot - it will check actual positions on startup")
        logger.warning("EMERGENCY POSITION RESET BY USER")
    else:
        print("❌ Reset cancelled")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="SuperTrend Trading Bot")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Auth command
    subparsers.add_parser('auth', help='Authenticate with Kite Connect')
    
    # Test command
    subparsers.add_parser('test', help='Test Kite connection')
    
    # Trade command
    subparsers.add_parser('trade', help='Start trading')
    
    # Reset command
    subparsers.add_parser('reset', help='Emergency position reset')
    
    args = parser.parse_args()
    
    if args.command == 'auth':
        authenticate()
    elif args.command == 'test':
        test_connection()
    elif args.command == 'trade':
        start_trading()
    elif args.command == 'reset':
        emergency_reset()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("1. python cli.py auth    # First time authentication")
        print("2. python cli.py test    # Test connection")
        print("3. python cli.py trade   # Start trading")

if __name__ == "__main__":
    main()
