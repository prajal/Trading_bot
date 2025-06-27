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
    Comprehensive market data validation system
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
    
    def validate_ohlc_data(self, df: pd.DataFrame, strict_mode: bool = True) -> bool:
        """
        Comprehensive OHLC data validation
        
        Args:
            df: DataFrame with OHLC data
            strict_mode: If True, fails on any validation error
            
        Returns:
            bool: True if data passes validation
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
            
            validation_results = []
            
            # 1. Basic OHLC relationship validation
            validation_results.append(self._validate_ohlc_relationships(df))
            
            # 2. Price range validation
            validation_results.append(self._validate_price_ranges(df))
            
            # 3. Price change validation
            validation_results.append(self._validate_price_changes(df))
            
            # 4. Volume validation (if present)
            if 'volume' in df.columns:
                validation_results.append(self._validate_volume_data(df))
            
            # 5. Gap analysis
            validation_results.append(self._validate_price_gaps(df))
            
            # 6. Data continuity validation
            validation_results.append(self._validate_data_continuity(df))
            
            # 7. Statistical outlier detection
            validation_results.append(self._validate_statistical_outliers(df))
            
            # Calculate overall validation score
            passed_validations = sum(validation_results)
            total_validations = len(validation_results)
            validation_score = passed_validations / total_validations
            
            # Log validation results
            logger.debug(f"Data validation score: {validation_score:.2%} ({passed_validations}/{total_validations})")
            
            if strict_mode:
                return validation_score == 1.0
            else:
                return validation_score >= self.quality_thresholds['acceptable']
        
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
        """Validate price ranges are within reasonable bounds"""
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
                
                # Check for NaN or infinite values
                invalid_values = (~df[col].isfinite()).sum()
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
                    
                    # In strict mode, this might be acceptable for volatile markets
                    # So we log but don't fail validation
            
            # Check for flat prices (potential data feed issues)
            for col in ['high', 'low']:
                flat_prices = (df[col] == df[col].shift(1)).sum()
                if flat_prices > len(df) * 0.1:  # More than 10% flat prices
                    logger.warning(f"High number of flat prices in {col}: {flat_prices}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price changes: {e}")
            return False
    
    def _validate_volume_data(self, df: pd.DataFrame) -> bool:
        """Validate volume data"""
        try:
            if 'volume' not in df.columns:
                return True
            
            # Check for negative volumes
            negative_volumes = (df['volume'] < 0).sum()
            if negative_volumes > 0:
                logger.error(f"Found {negative_volumes} negative volume values")
                return False
            
            # Check for zero volumes (might be acceptable for some timeframes)
            zero_volumes = (df['volume'] == 0).sum()
            if zero_volumes > len(df) * 0.05:  # More than 5% zero volumes
                logger.warning(f"High number of zero volume candles: {zero_volumes}")
            
            # Check for extremely high volumes (potential data errors)
            median_volume = df['volume'].median()
            if median_volume > 0:
                extreme_volumes = (df['volume'] > median_volume * 100).sum()
                if extreme_volumes > 0:
                    logger.warning(f"Found {extreme_volumes} extremely high volume candles")
            
            # Check for NaN values
            nan_volumes = df['volume'].isna().sum()
            if nan_volumes > 0:
                logger.warning(f"Found {nan_volumes} NaN volume values")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating volume data: {e}")
            return False
    
    def _validate_price_gaps(self, df: pd.DataFrame) -> bool:
        """Validate price gaps between consecutive candles"""
        try:
            if len(df) < 2:
                return True
            
            # Calculate gaps between consecutive closes and next opens
            # Note: This assumes continuous trading; adjust for market hours if needed
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
                
                # Large gaps might be normal for some instruments, so we don't fail validation
            
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
            if df.index.name == 'date' or 'date' in df.index.names:
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
            
            # Check data density (for minute data, should be relatively continuous)
            # This is a simplified check and might need adjustment based on market hours
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
    
    def _validate_statistical_outliers(self, df: pd.DataFrame) -> bool:
        """Detect statistical outliers in price data"""
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
            
            if outlier_percentage > 0.05:  # More than 5% outliers
                logger.warning(f"High number of statistical outliers: {outliers} ({outlier_percentage:.1%})")
            
            # This doesn't fail validation as outliers can be normal in volatile markets
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
            
            # 3. Price stability score (fewer extreme changes = higher score)
            if len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                extreme_changes = (price_changes > 0.1).sum()  # More than 10% change
                stability_score = max(0, 1 - (extreme_changes / len(df)))
                quality_metrics['price_stability'] = stability_score
                
                if stability_score < 0.8:
                    issues.append(f"High price volatility detected")
            else:
                quality_metrics['price_stability'] = 1.0
            
            # 4. Volume consistency (if available)
            if 'volume' in df.columns:
                zero_volumes = (df['volume'] == 0).sum()
                volume_consistency = max(0, 1 - (zero_volumes / len(df)))
                quality_metrics['volume_consistency'] = volume_consistency
                
                if volume_consistency < 0.9:
                    issues.append("Volume data inconsistencies found")
            else:
                quality_metrics['volume_consistency'] = 1.0
            
            # 5. Temporal consistency
            if hasattr(df.index, 'is_monotonic_increasing'):
                temporal_consistency = 1.0 if df.index.is_monotonic_increasing else 0.5
            else:
                temporal_consistency = 1.0
            
            quality_metrics['temporal_consistency'] = temporal_consistency
            
            if temporal_consistency < 1.0:
                issues.append("Temporal ordering issues detected")
            
            # Calculate overall score (weighted average)
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
            if overall_score >= self.quality_thresholds['excellent']:
                quality_rating = 'excellent'
            elif overall_score >= self.quality_thresholds['good']:
                quality_rating = 'good'
            elif overall_score >= self.quality_thresholds['acceptable']:
                quality_rating = 'acceptable'
            elif overall_score >= self.quality_thresholds['poor']:
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
    
    def clean_data(self, df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Clean and fix common data issues
        
        Args:
            df: Input DataFrame
            aggressive: If True, applies more aggressive cleaning
            
        Returns:
            Cleaned DataFrame
        """
        try:
            if df.empty:
                return df
            
            df_clean = df.copy()
            cleaning_actions = []
            
            # 1. Remove rows with invalid OHLC relationships
            initial_count = len(df_clean)
            
            # Remove rows where high < low (definitely invalid)
            invalid_hl = df_clean['high'] < df_clean['low']
            df_clean = df_clean[~invalid_hl]
            if invalid_hl.sum() > 0:
                cleaning_actions.append(f"Removed {invalid_hl.sum()} rows with high < low")
            
            # Remove rows with negative or zero prices
            for col in ['open', 'high', 'low', 'close']:
                invalid_prices = df_clean[col] <= 0
                df_clean = df_clean[~invalid_prices]
                if invalid_prices.sum() > 0:
                    cleaning_actions.append(f"Removed {invalid_prices.sum()} rows with invalid {col} prices")
            
            # 2. Fix OHLC relationships where possible
            # If open/close are outside high/low range, adjust high/low
            df_clean['high'] = df_clean[['high', 'open', 'close']].max(axis=1)
            df_clean['low'] = df_clean[['low', 'open', 'close']].min(axis=1)
            
            # 3. Handle extreme price changes
            if aggressive and len(df_clean) > 1:
                # Calculate price changes
                price_change = df_clean['close'].pct_change().abs()
                
                # Cap extreme changes at 50%
                extreme_changes = price_change > 0.5
                if extreme_changes.sum() > 0:
                    # For extreme changes, use previous close as approximation
                    for idx in df_clean.index[extreme_changes]:
                        if idx > 0:
                            prev_close = df_clean.loc[df_clean.index[df_clean.index.get_loc(idx) - 1], 'close']
                            # Adjust all OHLC to be within 20% of previous close
                            max_price = prev_close * 1.2
                            min_price = prev_close * 0.8
                            
                            df_clean.loc[idx, 'high'] = min(df_clean.loc[idx, 'high'], max_price)
                            df_clean.loc[idx, 'low'] = max(df_clean.loc[idx, 'low'], min_price)
                            df_clean.loc[idx, 'open'] = max(min_price, min(df_clean.loc[idx, 'open'], max_price))
                            df_clean.loc[idx, 'close'] = max(min_price, min(df_clean.loc[idx, 'close'], max_price))
                    
                    cleaning_actions.append(f"Capped {extreme_changes.sum()} extreme price changes")
            
            # 4. Handle volume data
            if 'volume' in df_clean.columns:
                # Remove negative volumes
                negative_vol = df_clean['volume'] < 0
                df_clean = df_clean[~negative_vol]
                if negative_vol.sum() > 0:
                    cleaning_actions.append(f"Removed {negative_vol.sum()} rows with negative volume")
                
                # Fill zero volumes with median volume (if aggressive cleaning)
                if aggressive:
                    zero_volumes = df_clean['volume'] == 0
                    if zero_volumes.sum() > 0:
                        median_volume = df_clean['volume'][df_clean['volume'] > 0].median()
                        if pd.notna(median_volume):
                            df_clean.loc[zero_volumes, 'volume'] = median_volume
                            cleaning_actions.append(f"Filled {zero_volumes.sum()} zero volumes with median")
            
            # 5. Remove duplicate timestamps
            if df_clean.index.duplicated().sum() > 0:
                duplicates = df_clean.index.duplicated().sum()
                df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
                cleaning_actions.append(f"Removed {duplicates} duplicate timestamps")
            
            # 6. Sort by timestamp
            if not df_clean.index.is_monotonic_increasing:
                df_clean = df_clean.sort_index()
                cleaning_actions.append("Sorted data by timestamp")
            
            final_count = len(df_clean)
            removed_count = initial_count - final_count
            
            if cleaning_actions:
                logger.info(f"Data cleaning completed: {removed_count} rows removed, {len(cleaning_actions)} actions taken")
                for action in cleaning_actions:
                    logger.debug(f"  - {action}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            return df  # Return original data if cleaning fails
    
    def validate_real_time_price(self, current_price: float, previous_price: float, 
                                symbol: str = "Unknown") -> Tuple[bool, str]:
        """
        Validate real-time price data
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
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
            
            # Price change validation (if previous price available)
            if previous_price and previous_price > 0:
                price_change_pct = abs(current_price - previous_price) / previous_price * 100
                
                if price_change_pct > self.max_price_change_percent:
                    return False, f"Extreme price change: {price_change_pct:.1f}% for {symbol}"
            
            return True, "Price validation passed"
            
        except Exception as e:
            return False, f"Price validation error: {e}"
    
    def validate_signal_data(self, signal_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate trading signal data
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        try:
            issues = []
            
            # Check required fields
            required_fields = ['close', 'supertrend', 'direction']
            for field in required_fields:
                if field not in signal_data:
                    issues.append(f"Missing required field: {field}")
                elif signal_data[field] is None:
                    issues.append(f"Field {field} is None")
            
            # Validate price data
            if 'close' in signal_data:
                price = signal_data['close']
                if not isinstance(price, (int, float)) or price <= 0:
                    issues.append(f"Invalid close price: {price}")
            
            if 'supertrend' in signal_data:
                st_value = signal_data['supertrend']
                if not isinstance(st_value, (int, float)) or st_value <= 0:
                    issues.append(f"Invalid SuperTrend value: {st_value}")
            
            # Validate direction
            if 'direction' in signal_data:
                direction = signal_data['direction']
                if direction not in [-1, 1]:
                    issues.append(f"Invalid direction: {direction} (must be -1 or 1)")
            
            # Validate confidence (if present)
            if 'confidence' in signal_data:
                confidence = signal_data['confidence']
                if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                    issues.append(f"Invalid confidence: {confidence} (must be between 0 and 1)")
            
            # Validate trend consistency
            if 'close' in signal_data and 'supertrend' in signal_data and 'direction' in signal_data:
                close_price = signal_data['close']
                supertrend = signal_data['supertrend']
                direction = signal_data['direction']
                
                # Check if price/SuperTrend relationship matches direction
                if direction == 1 and close_price < supertrend:
                    issues.append("Inconsistent signal: Direction is UP but price is below SuperTrend")
                elif direction == -1 and close_price > supertrend:
                    issues.append("Inconsistent signal: Direction is DOWN but price is above SuperTrend")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Signal validation error: {e}"]
    
    def generate_data_quality_report(self, df: pd.DataFrame, symbol: str = "Unknown") -> str:
        """Generate comprehensive data quality report"""
        try:
            quality_score = self.calculate_data_quality_score(df)
            
            report = f"""
DATA QUALITY REPORT
==================
Symbol: {symbol}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
Overall Score: {quality_score['overall_score']:.2%}
Quality Rating: {quality_score['quality_rating'].upper()}
Data Points: {quality_score.get('data_points', 0)}

DETAILED METRICS
---------------
"""
            
            if 'detailed_metrics' in quality_score:
                for metric, score in quality_score['detailed_metrics'].items():
                    report += f"{metric.replace('_', ' ').title()}: {score:.2%}\n"
            
            if quality_score.get('issues'):
                report += f"\nISSUES FOUND\n-----------\n"
                for i, issue in enumerate(quality_score['issues'], 1):
                    report += f"{i}. {issue}\n"
            
            if quality_score.get('time_range'):
                time_range = quality_score['time_range']
                if time_range['start'] and time_range['end']:
                    report += f"\nTIME RANGE\n----------\n"
                    report += f"Start: {time_range['start']}\n"
                    report += f"End: {time_range['end']}\n"
                    
                    # Calculate duration
                    if hasattr(time_range['start'], 'strftime'):
                        duration = time_range['end'] - time_range['start']
                        report += f"Duration: {duration}\n"
            
            # Add recommendations
            report += f"\nRECOMMENDATIONS\n--------------\n"
            
            if quality_score['overall_score'] >= 0.95:
                report += "✅ Data quality is excellent. No action required.\n"
            elif quality_score['overall_score'] >= 0.85:
                report += "✅ Data quality is good. Monitor for any degradation.\n"
            elif quality_score['overall_score'] >= 0.70:
                report += "⚠️  Data quality is acceptable but could be improved.\n"
                report += "   Consider implementing data cleaning procedures.\n"
            else:
                report += "❌ Data quality is poor. Immediate action required.\n"
                report += "   Recommend thorough data cleaning before use.\n"
                report += "   Consider switching data sources if issues persist.\n"
            
            return report
            
        except Exception as e:
            return f"Error generating data quality report: {e}"
