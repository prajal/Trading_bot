import json
import os
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException
from config.enhanced_settings import Settings, ConfigurationError
from utils.enhanced_logger import get_logger
import time
from typing import Optional, Dict, Any

logger = get_logger(__name__)

class AuthenticationError(Exception):
    """Authentication related errors"""
    pass

class KiteAuth:
    """Enhanced Kite Connect authentication with comprehensive error handling"""
    
    def __init__(self):
        try:
            Settings.validate_credentials()
            self.api_key = Settings.get_kite_api_key()
            self.api_secret = Settings.get_kite_api_secret()
        except ConfigurationError as e:
            logger.error(f"Configuration error during auth initialization: {e}")
            raise AuthenticationError(f"Invalid configuration: {e}")
        
        self.token_file = Settings.TOKEN_FILE
        self.kite: Optional[KiteConnect] = None
        self._connection_validated = False
        self._last_validation_time: Optional[datetime] = None
        self._validation_interval = timedelta(minutes=30)  # Re-validate every 30 minutes
        
        # Connection retry settings
        self.max_connection_retries = 3
        self.connection_retry_delay = 2.0
        
        logger.info("Enhanced KiteAuth initialized")
    
    def generate_login_url(self) -> str:
        """Generate login URL for authentication"""
        try:
            import urllib.parse
            login_url = (f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"
                        f"&redirect_uri={urllib.parse.quote_plus(Settings.KITE_REDIRECT_URI)}")
            logger.info("Login URL generated successfully")
            return login_url
        except Exception as e:
            logger.error(f"Error generating login URL: {e}")
            raise AuthenticationError(f"Failed to generate login URL: {e}")
    
    def create_session(self, request_token: str) -> bool:
        """Create session using request token with comprehensive error handling"""
        if not request_token or not request_token.strip():
            raise AuthenticationError("Request token cannot be empty")
        
        try:
            kite = KiteConnect(api_key=self.api_key)
            
            logger.info("Creating new Kite session...")
            data = kite.generate_session(request_token.strip(), api_secret=self.api_secret)
            
            # Validate session data
            if not data.get('access_token'):
                raise AuthenticationError("No access token received from Kite")
            
            # Save tokens with enhanced error handling
            Settings.ensure_directories()
            tokens = {
                "access_token": data["access_token"],
                "public_token": data["public_token"],
                "refresh_token": data.get("refresh_token", ""),
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=1)).isoformat(),
                "api_key": self.api_key,  # Store for validation
                "user_id": data.get("user_id", ""),
                "user_name": data.get("user_name", "")
            }
            
            # Create backup before saving new tokens
            self._backup_existing_tokens()
            
            with open(self.token_file, 'w') as f:
                json.dump(tokens, f, indent=2)
            
            # Validate the new session immediately
            kite.set_access_token(data["access_token"])
            profile = kite.profile()
            
            logger.info(f"Session created successfully for user: {profile.get('user_name', 'Unknown')}")
            
            # Reset validation state
            self._connection_validated = True
            self._last_validation_time = datetime.now()
            
            return True
            
        except TokenException as e:
            logger.error(f"Token error during session creation: {e}")
            raise AuthenticationError(f"Invalid request token: {e}")
        except NetworkException as e:
            logger.error(f"Network error during session creation: {e}")
            raise AuthenticationError(f"Network error: {e}. Please check your internet connection.")
        except KiteException as e:
            logger.error(f"Kite API error during session creation: {e}")
            raise AuthenticationError(f"Kite API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during session creation: {e}")
            raise AuthenticationError(f"Unexpected error: {e}")
    
    def _backup_existing_tokens(self):
        """Backup existing tokens before creating new ones"""
        try:
            if self.token_file.exists():
                backup_file = Settings.BACKUP_DIR / f"tokens_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                Settings.ensure_directories()
                
                with open(self.token_file, 'r') as src:
                    with open(backup_file, 'w') as dst:
                        dst.write(src.read())
                
                logger.debug(f"Token backup created: {backup_file}")
        except Exception as e:
            logger.warning(f"Could not backup existing tokens: {e}")
    
    def _is_token_expired(self, tokens: Dict[str, Any]) -> bool:
        """Check if token is expired"""
        try:
            expires_at_str = tokens.get('expires_at')
            if not expires_at_str:
                # No expiry info, assume it's old format and might be expired
                created_at_str = tokens.get('created_at')
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    # Tokens typically expire after 24 hours
                    return datetime.now() - created_at > timedelta(hours=23)
                return True  # No creation time, assume expired
            
            expires_at = datetime.fromisoformat(expires_at_str)
            return datetime.now() >= expires_at
            
        except Exception as e:
            logger.warning(f"Error checking token expiry: {e}")
            return True  # Assume expired if we can't determine
    
    def _validate_connection(self, force: bool = False) -> bool:
        """Validate current connection"""
        try:
            # Check if we need to revalidate
            if (not force and self._connection_validated and 
                self._last_validation_time and 
                datetime.now() - self._last_validation_time < self._validation_interval):
                return True
            
            if not self.kite:
                return False
            
            # Test connection with API call
            profile = self.kite.profile()
            margins = self.kite.margins()
            
            # Validate response
            if not profile or not margins:
                logger.warning("Invalid response from Kite API during validation")
                return False
            
            logger.debug("Connection validation successful")
            self._connection_validated = True
            self._last_validation_time = datetime.now()
            return True
            
        except TokenException as e:
            logger.warning(f"Token validation failed: {e}")
            self._connection_validated = False
            return False
        except NetworkException as e:
            logger.warning(f"Network error during validation: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            self._connection_validated = False
            return False
    
    def get_kite_instance(self) -> Optional[KiteConnect]:
        """Get authenticated Kite instance with comprehensive error handling"""
        # Try to use existing instance if valid
        if self.kite and self._validate_connection():
            return self.kite
        
        # Load and validate tokens
        try:
            if not self.token_file.exists():
                logger.error("No token file found. Please authenticate first.")
                return None
            
            with open(self.token_file, 'r') as f:
                tokens = json.load(f)
            
            # Validate token structure
            required_fields = ['access_token', 'api_key']
            missing_fields = [field for field in required_fields if field not in tokens]
            if missing_fields:
                logger.error(f"Token file missing required fields: {missing_fields}")
                return None
            
            # Check if token is for current API key
            if tokens.get('api_key') != self.api_key:
                logger.error("Token file is for a different API key")
                return None
            
            # Check if token is expired
            if self._is_token_expired(tokens):
                logger.warning("Access token has expired. Please re-authenticate.")
                return None
            
            # Create Kite instance and set token
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(tokens['access_token'])
            
            # Validate connection with retries
            for attempt in range(self.max_connection_retries):
                try:
                    if self._validate_connection(force=True):
                        profile = self.kite.profile()
                        logger.info(f"Connected to Kite as: {profile.get('user_name', 'Unknown')}")
                        return self.kite
                    else:
                        if attempt < self.max_connection_retries - 1:
                            logger.warning(f"Connection validation failed, retrying in {self.connection_retry_delay}s...")
                            time.sleep(self.connection_retry_delay)
                            continue
                        else:
                            logger.error("Connection validation failed after all retries")
                            return None
                            
                except Exception as e:
                    logger.error(f"Error during connection attempt {attempt + 1}: {e}")
                    if attempt < self.max_connection_retries - 1:
                        time.sleep(self.connection_retry_delay)
                        continue
                    else:
                        return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in token file: {e}")
            return None
        except FileNotFoundError:
            logger.error("Token file not found")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading tokens: {e}")
            return None
        
        return None
    
    def test_connection(self) -> bool:
        """Test Kite connection with detailed validation"""
        try:
            kite = self.get_kite_instance()
            if not kite:
                logger.error("Failed to get Kite instance for connection test")
                return False
            
            # Comprehensive connection test
            logger.info("Testing Kite connection...")
            
            # Test 1: Profile
            profile = kite.profile()
            if not profile:
                logger.error("Failed to fetch profile")
                return False
            
            # Test 2: Margins
            margins = kite.margins()
            if not margins:
                logger.error("Failed to fetch margins")
                return False
            
            # Test 3: Instruments (sample check)
            try:
                instruments = kite.instruments("NSE")
                if not instruments or not isinstance(instruments, list):
                    logger.warning("Instruments data seems invalid")
                else:
                    logger.debug(f"Instruments check passed: {len(instruments)} instruments available")
            except Exception as e:
                logger.warning(f"Instruments check failed (non-critical): {e}")
            
            logger.info("Connection test successful")
            logger.info(f"User: {profile.get('user_name', 'Unknown')}")
            logger.info(f"Broker: {profile.get('broker', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed account information with error handling"""
        try:
            kite = self.get_kite_instance()
            if not kite:
                return None
            
            profile = kite.profile()
            margins = kite.margins()
            
            if not profile or not margins:
                logger.error("Failed to fetch account information")
                return None
            
            # Extract equity margins safely
            equity = margins.get('equity', {})
            if isinstance(equity, dict):
                available_cash = equity.get('available', {}).get('cash', 0) if isinstance(equity.get('available'), dict) else 0
                net_worth = equity.get('net', 0)
            else:
                available_cash = 0
                net_worth = 0
            
            account_info = {
                'user_name': profile.get('user_name', 'Unknown'),
                'user_id': profile.get('user_id', 'Unknown'),
                'broker': profile.get('broker', 'Unknown'),
                'email': profile.get('email', 'Not provided'),
                'available_cash': float(available_cash),
                'net_worth': float(net_worth),
                'account_status': 'Active',  # Assume active if we can fetch data
                'last_updated': datetime.now().isoformat()
            }
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return None
    
    def invalidate_token(self):
        """Invalidate current token and clean up"""
        try:
            if self.kite:
                try:
                    self.kite.invalidate_access_token()
                    logger.info("Token invalidated successfully")
                except Exception as e:
                    logger.warning(f"Error invalidating token (non-critical): {e}")
            
            # Backup current tokens before deletion
            self._backup_existing_tokens()
            
            if self.token_file.exists():
                self.token_file.unlink()
                logger.info("Token file removed")
            
            # Reset state
            self.kite = None
            self._connection_validated = False
            self._last_validation_time = None
            
        except Exception as e:
            logger.error(f"Error during token invalidation: {e}")
    
    def refresh_token_if_needed(self) -> bool:
        """Refresh token if it's close to expiry"""
        try:
            if not self.token_file.exists():
                return False
            
            with open(self.token_file, 'r') as f:
                tokens = json.load(f)
            
            # Check if token expires soon (within 2 hours)
            expires_at_str = tokens.get('expires_at')
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                time_to_expiry = expires_at - datetime.now()
                
                if time_to_expiry < timedelta(hours=2):
                    logger.warning("Token expires soon. Please re-authenticate.")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking token refresh: {e}")
            return False
    
    def get_connection_health(self) -> Dict[str, Any]:
        """Get detailed connection health information"""
        try:
            health = {
                'connected': False,
                'last_validation': None,
                'token_valid': False,
                'token_expires_at': None,
                'api_key_valid': False,
                'connection_retries_available': self.max_connection_retries,
                'last_error': None
            }
            
            # Check if we have a Kite instance
            if self.kite:
                health['connected'] = True
                
                # Check validation status
                if self._last_validation_time:
                    health['last_validation'] = self._last_validation_time.isoformat()
                
                # Check token validity
                try:
                    if self.token_file.exists():
                        with open(self.token_file, 'r') as f:
                            tokens = json.load(f)
                        
                        health['token_valid'] = not self._is_token_expired(tokens)
                        
                        expires_at_str = tokens.get('expires_at')
                        if expires_at_str:
                            health['token_expires_at'] = expires_at_str
                        
                        health['api_key_valid'] = tokens.get('api_key') == self.api_key
                        
                except Exception as e:
                    health['last_error'] = f"Error checking token: {e}"
                
                # Test current connection
                try:
                    if self._validate_connection():
                        health['connection_status'] = 'healthy'
                    else:
                        health['connection_status'] = 'unhealthy'
                except Exception as e:
                    health['connection_status'] = 'error'
                    health['last_error'] = str(e)
            
            return health
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'connection_status': 'error'
            }
    
    def _get_new_session_interactive(self) -> bool:
        """Interactive session creation (for CLI use)"""
        try:
            logger.info("Starting interactive authentication process...")
            
            login_url = self.generate_login_url()
            print("\n🔐 Kite Connect Authentication Required")
            print("=" * 50)
            print("1. Visit the following URL:")
            print(f"   {login_url}")
            print("2. Login with your Zerodha credentials")
            print("3. Authorize the application")
            print("4. Copy the 'request_token' from the redirected URL")
            print("5. Paste it below\n")
            
            request_token = input("Enter request_token: ").strip()
            
            if not request_token:
                logger.error("No request token provided")
                return False
            
            return self.create_session(request_token)
            
        except KeyboardInterrupt:
            logger.info("Authentication cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Error in interactive session creation: {e}")
            return False
    
    def ensure_authentication(self, interactive: bool = True) -> bool:
        """Ensure we have valid authentication, prompt if needed"""
        try:
            # Try existing authentication first
            if self.test_connection():
                return True
            
            logger.info("No valid authentication found")
            
            if interactive:
                return self._get_new_session_interactive()
            else:
                logger.error("Authentication required but running in non-interactive mode")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring authentication: {e}")
            return False
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get current session information"""
        try:
            if not self.token_file.exists():
                return None
            
            with open(self.token_file, 'r') as f:
                tokens = json.load(f)
            
            session_info = {
                'user_name': tokens.get('user_name', 'Unknown'),
                'user_id': tokens.get('user_id', 'Unknown'),
                'created_at': tokens.get('created_at'),
                'expires_at': tokens.get('expires_at'),
                'api_key': tokens.get('api_key', '')[:8] + '...',  # Partial key for security
                'token_valid': not self._is_token_expired(tokens),
                'connection_healthy': self._validate_connection() if self.kite else False
            }
            
            return session_info
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return None

def main():
    """Main authentication flow for CLI usage"""
    try:
        print("🔐 Kite Connect Authentication Manager")
        print("=" * 50)
        
        auth = KiteAuth()
        
        # Check current authentication status
        session_info = auth.get_session_info()
        
        if session_info and session_info.get('token_valid'):
            print(f"✅ Already authenticated as: {session_info.get('user_name', 'Unknown')}")
            
            # Test connection
            if auth.test_connection():
                print("✅ Connection test passed")
                
                # Show account info
                account_info = auth.get_account_info()
                if account_info:
                    print(f"\n💼 Account Information:")
                    print(f"   Name: {account_info['user_name']}")
                    print(f"   Available Cash: ₹{account_info['available_cash']:,.2f}")
                    print(f"   Net Worth: ₹{account_info['net_worth']:,.2f}")
                    print(f"   Broker: {account_info['broker']}")
                
                return True
            else:
                print("❌ Connection test failed")
        
        # Need new authentication
        print("\n🔄 Starting authentication process...")
        
        if auth.ensure_authentication(interactive=True):
            print("✅ Authentication successful!")
            
            # Show account info
            account_info = auth.get_account_info()
            if account_info:
                print(f"\n💼 Account Information:")
                print(f"   Name: {account_info['user_name']}")
                print(f"   Available Cash: ₹{account_info['available_cash']:,.2f}")
                print(f"   Net Worth: ₹{account_info['net_worth']:,.2f}")
            
            return True
        else:
            print("❌ Authentication failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Authentication cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Authentication error: {e}")
        logger.error(f"Authentication error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)