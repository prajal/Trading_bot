# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2024-06-24

### Added
- Dynamic account balance setting via CLI
- Account information display on authentication
- Command-line amount override with `--amount` flag
- Trading preferences persistence
- Enhanced signal detection with direction tracking
- Signal deduplication to prevent repeated trades
- Comprehensive debug logging

### Changed
- Account balance is now input-driven instead of hardcoded
- Improved SuperTrend signal detection logic
- Enhanced position synchronization
- Better error handling and logging

### Fixed
- Fixed inverted SuperTrend direction logic
- Fixed signal detection for trend changes
- Resolved circular dependency in settings

## [1.0.0] - 2024-06-23

### Initial Release
- Basic SuperTrend strategy implementation
- Zerodha Kite Connect integration
- MIS leverage support
- Position synchronization
- Stop loss and risk management
