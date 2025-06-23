import json
import os
from datetime import datetime
from kiteconnect import KiteConnect
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

class KiteAuth:
    """Handles Kite Connect authentication"""
    
    def __init__(self):
        self.api_key = Settings.KITE_API_KEY
        self.api_secret = Settings.KITE_API_SECRET
        self.token_file = Settings.TOKEN_FILE
        self.kite = None
    
    def generate_login_url(self) -> str:
        """Generate login URL for authentication"""
        import urllib.parse
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}&redirect_uri={urllib.parse.quote_plus(Settings.KITE_REDIRECT_URI)}"
        return login_url
    
    def create_session(self, request_token: str) -> bool:
        """Create session using request token"""
        try:
            kite = KiteConnect(api_key=self.api_key)
            data = kite.generate_session(request_token, api_secret=self.api_secret)
            
            # Save tokens
            Settings.ensure_directories()
            tokens = {
                "access_token": data["access_token"],
                "public_token": data["public_token"],
                "refresh_token": data.get("refresh_token", ""),
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(tokens, f, indent=2)
            
            logger.info("Session created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def get_kite_instance(self) -> KiteConnect:
        """Get authenticated Kite instance"""
        if self.kite:
            return self.kite
        
        try:
            if not self.token_file.exists():
                logger.error("No token file found. Please authenticate first.")
                return None
            
            with open(self.token_file, 'r') as f:
                tokens = json.load(f)
            
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(tokens['access_token'])
            
            # Test connection
            profile = self.kite.profile()
            logger.info(f"Connected to Kite as: {profile.get('user_name', 'Unknown')}")
            
            return self.kite
            
        except Exception as e:
            logger.error(f"Failed to establish Kite session: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Kite connection"""
        kite = self.get_kite_instance()
        if not kite:
            return False
        
        try:
            profile = kite.profile()
            margins = kite.margins()
            logger.info("Connection test successful")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def invalidate_token(self):
        """Invalidate current token"""
        if self.kite:
            try:
                self.kite.invalidate_access_token()
                logger.info("Token invalidated successfully")
            except Exception as e:
                logger.warning(f"Error invalidating token: {e}")
        
        if self.token_file.exists():
            self.token_file.unlink()
            logger.info("Token file removed")
        
        self.kite = None

    def _get_new_session(self):
        """Generate a new Kite session and save tokens"""
        logger.info("Generating new Kite session...")
        
        login_url = self.kite.login_url()
        print("1. Visit:", login_url)
        print("2. Login and authorize the app")
        print("3. Copy the request_token from the redirected URL")
        request_token = input("Enter request_token: ")

        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.kite.set_access_token(data["access_token"])
            logger.info("✅ Session created successfully!")
            
            # Save tokens
            Settings.ensure_directories()
            tokens = {
                "access_token": data["access_token"],
                "public_token": data["public_token"],
                "refresh_token": data.get("refresh_token", ""),
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(tokens, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

def main():
    """Main authentication flow"""
    print("🔐 Kite Connect Authentication")
    print("========================================\n")
    
    auth = KiteAuth()
    
    # Try to get an existing session
    kite_instance = auth.get_kite_instance()
    
    if kite_instance:
        try:
            profile = kite_instance.profile()
            print(f"✅ Already authenticated as {profile.get('user_name')}")
            print(f"   Balance: ₹{profile.get('day').get('equity').get('available_margin')}")
            return
        except Exception:
            logger.warning("Existing token is invalid. Re-authenticating...")
    
    # If no valid session, start new authentication
    auth._get_new_session()

if __name__ == "__main__":
    main()
