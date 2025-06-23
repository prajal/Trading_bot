#!/usr/bin/env python3
"""
Startup Configuration Checker
============================

This script validates all configuration before starting the trading bot.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_configuration():
    """Check all configuration settings"""
    print("🔍 Checking Configuration...")
    
    try:
        from config.improved_settings import Settings
        from utils.validators import ValidationError
        from utils.exceptions import ConfigurationError
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Please run: python quick_improvements.py first")
        return False
    
    # Validate configuration
    errors = Settings.validate_configuration()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print("💡 Please fix these errors before starting trading")
        return False
    
    # Print configuration
    Settings.print_configuration()
    
    # Check directories
    try:
        Settings.ensure_directories()
        print("✅ Directories created/verified")
    except Exception as e:
        print(f"❌ Directory Error: {e}")
        return False
    
    # Check .env file
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("💡 Copy .env.example to .env and add your credentials")
        return False
    
    print("✅ Configuration check passed!")
    return True

def main():
    """Main function"""
    print("🎯 Trading Bot Configuration Checker")
    print("=" * 40)
    
    if check_configuration():
        print("\n🚀 Ready to start trading!")
        print("💡 Run: python cli.py test")
    else:
        print("\n❌ Configuration issues found")
        print("💡 Please fix the issues above before trading")
        sys.exit(1)

if __name__ == "__main__":
    main()
