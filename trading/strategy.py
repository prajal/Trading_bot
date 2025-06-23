import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class SuperTrendStrategy:
    """SuperTrend trading strategy - FIXED VERSION with Enhanced Signal Detection"""
    
    def __init__(self, atr_period: int = 10, factor: float = 3.0):
        self.atr_period = atr_period
        self.factor = factor
        self.last_direction = None  # Track last known direction
    
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator - CORRECTED VERSION"""
        if len(df) < self.atr_period + 10:
            raise ValueError(f"Need at least {self.atr_period + 10} candles")

        df = df.copy()
        
        # Calculate True Range (TR) - Standard method
        df['prev_close'] = df['close'].shift(1)
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['prev_close'])
        df['lc'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)

        # Calculate ATR using Exponential Moving Average (EMA) method
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        # Calculate basic upper and lower bands
        df['hl2'] = (df['high'] + df['low']) / 2  # Median price
        df['basic_upper'] = df['hl2'] + (self.factor * df['atr'])
        df['basic_lower'] = df['hl2'] - (self.factor * df['atr'])

        # Initialize final bands and SuperTrend
        df['final_upper'] = 0.0
        df['final_lower'] = 0.0
        df['supertrend'] = 0.0
        df['direction'] = 1  # 1 = uptrend (GREEN), -1 = downtrend (RED)

        for i in range(len(df)):
            if i == 0:
                # Initialize first values
                df.iloc[i, df.columns.get_loc('final_upper')] = df.iloc[i]['basic_upper']
                df.iloc[i, df.columns.get_loc('final_lower')] = df.iloc[i]['basic_lower']
                df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['basic_upper']
                df.iloc[i, df.columns.get_loc('direction')] = 1
            else:
                # Get previous values
                prev_final_upper = df.iloc[i-1]['final_upper']
                prev_final_lower = df.iloc[i-1]['final_lower']
                prev_close = df.iloc[i-1]['close']
                prev_supertrend = df.iloc[i-1]['supertrend']
                
                current_basic_upper = df.iloc[i]['basic_upper']
                current_basic_lower = df.iloc[i]['basic_lower']
                current_close = df.iloc[i]['close']

                # Calculate Final Upper Band
                if current_basic_upper < prev_final_upper or prev_close > prev_final_upper:
                    final_upper = current_basic_upper
                else:
                    final_upper = prev_final_upper

                # Calculate Final Lower Band
                if current_basic_lower > prev_final_lower or prev_close < prev_final_lower:
                    final_lower = current_basic_lower
                else:
                    final_lower = prev_final_lower

                df.iloc[i, df.columns.get_loc('final_upper')] = final_upper
                df.iloc[i, df.columns.get_loc('final_lower')] = final_lower

                # Calculate SuperTrend and Direction
                if (prev_supertrend == prev_final_upper and current_close < final_upper) or \
                   (prev_supertrend == prev_final_lower and current_close < final_lower):
                    # Downtrend - SuperTrend is upper band
                    supertrend = final_upper
                    direction = -1  # RED (Downtrend)
                else:
                    # Uptrend - SuperTrend is lower band
                    supertrend = final_lower
                    direction = 1   # GREEN (Uptrend)

                df.iloc[i, df.columns.get_loc('supertrend')] = supertrend
                df.iloc[i, df.columns.get_loc('direction')] = direction

        # Clean up temporary columns
        df.drop(['prev_close', 'hl', 'hc', 'lc', 'tr', 'hl2', 'basic_upper', 'basic_lower'], 
                axis=1, inplace=True)
        
        # Add trend description for logging
        df['trend_desc'] = df['direction'].apply(lambda x: "GREEN (Uptrend)" if x == 1 else "RED (Downtrend)")
        
        return df
    
    def get_signal(self, df: pd.DataFrame, has_position: bool = False) -> Tuple[str, dict]:
        """Get trading signal from SuperTrend - ENHANCED VERSION"""
        try:
            df_with_st = self.calculate_supertrend(df)
            
            if len(df_with_st) < 2:
                return "HOLD", {"error": "Insufficient data for signal generation"}
            
            # Current and previous direction
            current_direction = df_with_st["direction"].iloc[-1]
            previous_direction = df_with_st["direction"].iloc[-2]
            
            # Current price and SuperTrend value
            current_close = df_with_st["close"].iloc[-1]
            current_supertrend = df_with_st["supertrend"].iloc[-1]
            current_trend_desc = df_with_st["trend_desc"].iloc[-1]
            
            # DEBUG: Log direction values
            logger.debug(f"Signal Detection - Previous: {previous_direction}, Current: {current_direction}, Last Known: {self.last_direction}")
            
            signal_data = {
                "close": current_close,
                "supertrend": current_supertrend,
                "direction": current_direction,
                "previous_direction": previous_direction,
                "trend": current_trend_desc,
                "price_vs_supertrend": "Above" if current_close > current_supertrend else "Below"
            }
            
            # ENHANCED SIGNAL DETECTION
            # Check if we have a direction change
            direction_changed = False
            
            # If this is first run or direction actually changed
            if self.last_direction is None:
                self.last_direction = current_direction
            elif self.last_direction != current_direction:
                direction_changed = True
                logger.info(f"🔄 Direction Change Detected: {self.last_direction} → {current_direction}")
            
            # CORRECTED SIGNAL LOGIC with ENHANCED DETECTION
            # BUY: When trend changes from RED (downtrend) to GREEN (uptrend)
            if direction_changed and self.last_direction == -1 and current_direction == 1 and not has_position:
                logger.info(f"🟢 SuperTrend ENTRY signal: Trend changed from RED to GREEN")
                logger.info(f"   Price: ₹{current_close:.2f}, SuperTrend: ₹{current_supertrend:.2f}")
                self.last_direction = current_direction
                return "BUY", signal_data
            
            # Alternative BUY detection using consecutive candles
            elif previous_direction == -1 and current_direction == 1 and not has_position:
                logger.info(f"🟢 SuperTrend ENTRY signal (Alt): Trend changed from RED to GREEN")
                logger.info(f"   Price: ₹{current_close:.2f}, SuperTrend: ₹{current_supertrend:.2f}")
                self.last_direction = current_direction
                return "BUY", signal_data
            
            # SELL: When trend changes from GREEN (uptrend) to RED (downtrend)
            elif direction_changed and self.last_direction == 1 and current_direction == -1 and has_position:
                logger.info(f"🔴 SuperTrend EXIT signal: Trend changed from GREEN to RED")
                logger.info(f"   Price: ₹{current_close:.2f}, SuperTrend: ₹{current_supertrend:.2f}")
                self.last_direction = current_direction
                return "SELL", signal_data
            
            # Alternative SELL detection using consecutive candles
            elif previous_direction == 1 and current_direction == -1 and has_position:
                logger.info(f"🔴 SuperTrend EXIT signal (Alt): Trend changed from GREEN to RED")
                logger.info(f"   Price: ₹{current_close:.2f}, SuperTrend: ₹{current_supertrend:.2f}")
                self.last_direction = current_direction
                return "SELL", signal_data
            
            # Additional exit condition if we have a position and trend is RED
            elif has_position and current_direction == -1:
                logger.info(f"🔴 SuperTrend EXIT signal: In downtrend with position")
                self.last_direction = current_direction
                return "SELL", signal_data
            
            else:
                # Update last known direction
                self.last_direction = current_direction
                
                # Log current status for monitoring
                if current_direction == 1:
                    logger.debug(f"📈 Trend: GREEN (Uptrend) - Price: ₹{current_close:.2f}, SuperTrend: ₹{current_supertrend:.2f}")
                else:
                    logger.debug(f"📉 Trend: RED (Downtrend) - Price: ₹{current_close:.2f}, SuperTrend: ₹{current_supertrend:.2f}")
                
                return "HOLD", signal_data
                
        except Exception as e:
            logger.error(f"Error calculating SuperTrend signal: {e}")
            return "ERROR", {"error": str(e)}
    
    def validate_signal(self, df: pd.DataFrame) -> bool:
        """Validate that SuperTrend is working correctly"""
        try:
            df_with_st = self.calculate_supertrend(df)
            
            # Check if we have reasonable values
            if df_with_st['atr'].isna().any():
                logger.error("ATR contains NaN values")
                return False
            
            if df_with_st['supertrend'].isna().any():
                logger.error("SuperTrend contains NaN values")
                return False
            
            # Check if direction values are valid
            valid_directions = df_with_st['direction'].isin([1, -1]).all()
            if not valid_directions:
                logger.error("Invalid direction values found")
                return False
            
            # Log validation success
            signals_count = len(df_with_st[df_with_st['direction'].diff() != 0]) - 1
            logger.info(f"✅ SuperTrend validation passed. Found {signals_count} direction changes.")
            
            return True
            
        except Exception as e:
            logger.error(f"SuperTrend validation failed: {e}")
            return False