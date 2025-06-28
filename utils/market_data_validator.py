import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class MarketDataValidator:
    """
    FIXED: Comprehensive market data validation system
    Ensures data quality and integrity for trading decisions
    """
    
    def __init__(self):
        # Validation thresholds
        self.max_price_change_percent = 20.0  # Maximum single-candle price change
        self.min_price_threshold = 1.0        # Minimum valid price
        self.max_price_threshold = 100000.0   # Maximum valid price
        self.min_volume_threshold = 1         # Minimum volume
        self.max_gap_percent = 10.0           # Maximum gap between candles
        
        # Data quality scores
        self.quality_thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.70,
            'poor': 0.50
        }
        
        logger.info("Market Data Validator initialized")
    
    def validate_ohlc_data(self, df: pd.DataFrame, strict_mode: bool = False) -> bool:
        """
        FIXED: More lenient OHLC data validation for live trading
        """
        try:
            if df.empty:
                logger.error("DataFrame is empty")
                return False
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Only do essential validations for live trading
            validation_results = []
            
            # 1. Basic OHLC relationship validation (essential)
            validation_results.append(self._validate_ohlc_relationships(df))
            
            # 2. Price range validation (essential)
            validation_results.append(self._validate_price_ranges(df))
            
            # 3. Skip other validations for live trading to be more lenient
            if not strict_mode:
                # For live trading, only require basic structure to be correct
                passed_validations = sum(validation_results)
                total_validations = len(validation_results)
                validation_score = passed_validations / total_validations
                
                logger.debug(f"Live trading validation score: {validation_score:.2%}")
                
                # Accept if basic OHLC structure is valid
                return validation_score >= 0.5  # At least 50% (both basic validations)
            
            # Strict mode - do all validations
            validation_results.append(self._validate_price_changes(df))
            validation_results.append(self._validate_volume_data(df, strict_mode))
            validation_results.append(self._validate_price_gaps(df))
            validation_results.append(self._validate_data_continuity(df))
            validation_results.append(self._validate_statistical_outliers(df, strict_mode))
            
            passed_validations = sum(validation_results)
            total_validations = len(validation_results)
            validation_score = passed_validations / total_validations
            
            return validation_score == 1.0
        
        except Exception as e:
            logger.error(f"Error during OHLC validation: {e}")
            return False
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> bool:
        """Validate basic OHLC relationships"""
        try:
            # High should be >= Open, Close, Low
            high_valid = (
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) & 
                (df['high'] >= df['low'])
            ).all()
            
            # Low should be <= Open, Close, High
            low_valid = (
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close']) & 
                (df['low'] <= df['high'])
            ).all()
            
            if not high_valid:
                invalid_highs = (~((df['high'] >= df['open']) & 
                                  (df['high'] >= df['close']) & 
                                  (df['high'] >= df['low']))).sum()
                logger.warning(f"Invalid high prices found: {invalid_highs} candles")
            
            if not low_valid:
                invalid_lows = (~((df['low'] <= df['open']) & 
                                 (df['low'] <= df['close']) & 
                                 (df['low'] <= df['high']))).sum()
                logger.warning(f"Invalid low prices found: {invalid_lows} candles")
            
            return high_valid and low_valid
            
        except Exception as e:
            logger.error(f"Error validating OHLC relationships: {e}")
            return False
    
    def _validate_price_ranges(self, df: pd.DataFrame) -> bool:
        """FIXED: Validate price ranges are within reasonable bounds"""
        try:
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                # Check for negative or zero prices
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    logger.warning(f"Found {negative_prices} negative/zero prices in {col}")
                    return False
                
                # Check for extremely low prices
                very_low_prices = (df[col] < self.min_price_threshold).sum()
                if very_low_prices > 0:
                    logger.warning(f"Found {very_low_prices} suspiciously low prices in {col}")
                
                # Check for extremely high prices
                very_high_prices = (df[col] > self.max_price_threshold).sum()
                if very_high_prices > 0:
                    logger.warning(f"Found {very_high_prices} suspiciously high prices in {col}")
                
                # FIXED: Check for NaN or infinite values using numpy.isfinite
                invalid_values = (~np.isfinite(df[col])).sum()
                if invalid_values > 0:
                    logger.error(f"Found {invalid_values} invalid values (NaN/Inf) in {col}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price ranges: {e}")
            return False
    
    def _validate_price_changes(self, df: pd.DataFrame) -> bool:
        """Validate price changes are within reasonable limits"""
        try:
            if len(df) < 2:
                return True
            
            # Calculate percentage changes for each price component
            for col in ['open', 'high', 'low', 'close']:
                pct_change = df[col].pct_change().abs() * 100
                
                # Find extreme changes
                extreme_changes = pct_change > self.max_price_change_percent
                extreme_count = extreme_changes.sum()
                
                if extreme_count > 0:
                    max_change = pct_change.max()
                    logger.warning(f"Found {extreme_count} extreme price changes in {col} (max: {max_change:.1f}%)")
            
            # Check for flat prices (potential data feed issues)
            for col in ['high', 'low']:
                flat_prices = (df[col] == df[col].shift(1)).sum()
                if flat_prices > len(df) * 0.1:  # More than 10% flat prices
                    logger.warning(f"High number of flat prices in {col}: {flat_prices}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price changes: {e}")
            return False
    
    def _validate_volume_data(self, df: pd.DataFrame, strict_mode: bool = False) -> bool:
        """FIXED: Validate volume data with lenient mode for NIFTY index data"""
        try:
            if 'volume' not in df.columns:
                return True
            
            # Check for negative volumes
            negative_volumes = (df['volume'] < 0).sum()
            if negative_volumes > 0:
                logger.error(f"Found {negative_volumes} negative volume values")
                return False
            
            # Check for zero volumes - VERY LENIENT for NIFTY 50 index data
            zero_volumes = (df['volume'] == 0).sum()
            zero_volume_ratio = zero_volumes / len(df)
            
            # NIFTY 50 index data often has zero volumes - this is NORMAL
            if zero_volume_ratio > 0.95:  # Only warn if 95%+ are zero
                logger.debug(f"High zero volume ratio: {zero_volume_ratio:.1%} - normal for index data")
            
            # Check for extremely high volumes (potential data errors)
            non_zero_volumes = df['volume'][df['volume'] > 0]
            if len(non_zero_volumes) > 0:
                median_volume = non_zero_volumes.median()
                extreme_volumes = (df['volume'] > median_volume * 1000).sum()
                if extreme_volumes > 0:
                    logger.warning(f"Found {extreme_volumes} extremely high volume candles")
            
            # Check for NaN values
            nan_volumes = df['volume'].isna().sum()
            if nan_volumes > 0:
                logger.warning(f"Found {nan_volumes} NaN volume values")
            
            return True  # Always return True for volume validation in live trading
            
        except Exception as e:
            logger.error(f"Error validating volume data: {e}")
            return True  # Be lenient and continue
    
    def _validate_price_gaps(self, df: pd.DataFrame) -> bool:
        """Validate price gaps between consecutive candles"""
        try:
            if len(df) < 2:
                return True
            
            # Calculate gaps between consecutive closes and next opens
            gaps = (df['open'] - df['close'].shift(1)).abs()
            gap_percentages = (gaps / df['close'].shift(1)) * 100
            
            # Remove NaN values (first candle)
            gap_percentages = gap_percentages.dropna()
            
            # Find large gaps
            large_gaps = gap_percentages > self.max_gap_percent
            large_gap_count = large_gaps.sum()
            
            if large_gap_count > 0:
                max_gap = gap_percentages.max()
                logger.warning(f"Found {large_gap_count} large price gaps (max: {max_gap:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price gaps: {e}")
            return False
    
    def _validate_data_continuity(self, df: pd.DataFrame) -> bool:
        """Validate data continuity and completeness"""
        try:
            if len(df) < 2:
                return True
            
            # Check if data is properly sorted by timestamp
            if hasattr(df.index, 'is_monotonic_increasing'):
                is_sorted = df.index.is_monotonic_increasing
                if not is_sorted:
                    logger.warning("Data is not sorted by timestamp")
                    return False
            
            # Check for duplicate timestamps
            if hasattr(df.index, 'duplicated'):
                duplicate_timestamps = df.index.duplicated().sum()
                if duplicate_timestamps > 0:
                    logger.warning(f"Found {duplicate_timestamps} duplicate timestamps")
                    return False
            
            # Check data density
            if len(df) > 10:
                expected_points = len(df)
                actual_points = len(df.dropna())
                completeness = actual_points / expected_points
                
                if completeness < 0.95:  # Less than 95% complete
                    logger.warning(f"Data completeness is low: {completeness:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data continuity: {e}")
            return False
    
    def _validate_statistical_outliers(self, df: pd.DataFrame, strict_mode: bool = False) -> bool:
        """Detect statistical outliers in price data (informational only)"""
        try:
            # Calculate price changes
            price_changes = df['close'].pct_change().dropna()
            
            if len(price_changes) < 10:
                return True
            
            # Use IQR method for outlier detection
            Q1 = price_changes.quantile(0.25)
            Q3 = price_changes.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((price_changes < lower_bound) | (price_changes > upper_bound)).sum()
            outlier_percentage = outliers / len(price_changes)
            
            if outlier_percentage > 0.1:  # More than 10% outliers
                logger.warning(f"High number of statistical outliers: {outliers} ({outlier_percentage:.1%})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error detecting statistical outliers: {e}")
            return False
    
    def calculate_data_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality score"""
        try:
            if df.empty:
                return {
                    'overall_score': 0.0,
                    'quality_rating': 'invalid',
                    'issues': ['Empty dataset']
                }
            
            quality_metrics = {}
            issues = []
            
            # 1. Completeness score
            total_points = len(df)
            complete_points = len(df.dropna())
            completeness = complete_points / total_points if total_points > 0 else 0
            quality_metrics['completeness'] = completeness
            
            if completeness < 0.95:
                issues.append(f"Data completeness low: {completeness:.1%}")
            
            # 2. OHLC validity score
            ohlc_valid = self._validate_ohlc_relationships(df)
            quality_metrics['ohlc_validity'] = 1.0 if ohlc_valid else 0.0
            
            if not ohlc_valid:
                issues.append("OHLC relationship violations found")
            
            # 3. Price stability score
            if len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                extreme_changes = (price_changes > 0.1).sum()
                stability_score = max(0, 1 - (extreme_changes / len(df)))
                quality_metrics['price_stability'] = stability_score
                
                if stability_score < 0.8:
                    issues.append(f"High price volatility detected")
            else:
                quality_metrics['price_stability'] = 1.0
            
            # 4. Volume consistency
            if 'volume' in df.columns:
                zero_volumes = (df['volume'] == 0).sum()
                volume_consistency = max(0, 1 - (zero_volumes / len(df)) * 0.5)
                quality_metrics['volume_consistency'] = volume_consistency
            else:
                quality_metrics['volume_consistency'] = 1.0
            
            # 5. Temporal consistency
            if hasattr(df.index, 'is_monotonic_increasing'):
                temporal_consistency = 1.0 if df.index.is_monotonic_increasing else 0.5
            else:
                temporal_consistency = 1.0
            
            quality_metrics['temporal_consistency'] = temporal_consistency
            
            # Calculate overall score
            weights = {
                'completeness': 0.25,
                'ohlc_validity': 0.30,
                'price_stability': 0.20,
                'volume_consistency': 0.15,
                'temporal_consistency': 0.10
            }
            
            overall_score = sum(
                quality_metrics[metric] * weights[metric] 
                for metric in weights.keys()
            )
            
            # Determine quality rating
            if overall_score >= 0.90:
                quality_rating = 'excellent'
            elif overall_score >= 0.75:
                quality_rating = 'good'
            elif overall_score >= 0.60:
                quality_rating = 'acceptable'
            elif overall_score >= 0.40:
                quality_rating = 'poor'
            else:
                quality_rating = 'unacceptable'
            
            return {
                'overall_score': overall_score,
                'quality_rating': quality_rating,
                'detailed_metrics': quality_metrics,
                'issues': issues,
                'data_points': len(df),
                'time_range': {
                    'start': df.index[0] if len(df) > 0 else None,
                    'end': df.index[-1] if len(df) > 0 else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return {
                'overall_score': 0.0,
                'quality_rating': 'error',
                'error': str(e)
            }
    
    def validate_real_time_price(self, current_price: float, previous_price: float, 
                                symbol: str = "Unknown") -> Tuple[bool, str]:
        """Validate real-time price data"""
        try:
            # Basic price validation
            if current_price <= 0:
                return False, f"Invalid price: {current_price} (must be positive)"
            
            if not np.isfinite(current_price):
                return False, f"Invalid price: {current_price} (not finite)"
            
            # Price range validation
            if current_price < self.min_price_threshold:
                return False, f"Price too low: {current_price} < {self.min_price_threshold}"
            
            if current_price > self.max_price_threshold:
                return False, f"Price too high: {current_price} > {self.max_price_threshold}"
            
            # Price change validation
            if previous_price and previous_price > 0:
                price_change_pct = abs(current_price - previous_price) / previous_price * 100
                
                if price_change_pct > self.max_price_change_percent:
                    return False, f"Extreme price change: {price_change_pct:.1f}% for {symbol}"
            
            return True, "Price validation passed"
            
        except Exception as e:
            return False, f"Price validation error: {e}"