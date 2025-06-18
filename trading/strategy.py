import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class SuperTrendStrategy:
    """SuperTrend trading strategy"""
    
    def __init__(self, atr_period: int = 10, factor: float = 3.0):
        self.atr_period = atr_period
        self.factor = factor
    
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator"""
        if len(df) < self.atr_period + 10:
            raise ValueError(f"Need at least {self.atr_period + 10} candles")

        df = df.copy()
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['high_low'] = df['high'] - df['low']
        df['high_prev_close'] = abs(df['high'] - df['prev_close'])
        df['low_prev_close'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

        # Calculate ATR using RMA (Rolling Moving Average)
        df['atr'] = 0.0
        for i in range(len(df)):
            if i == 0:
                df.iloc[i, df.columns.get_loc('atr')] = df.iloc[i]['tr']
            else:
                prev_atr = df.iloc[i-1]['atr']
                current_tr = df.iloc[i]['tr']
                df.iloc[i, df.columns.get_loc('atr')] = (prev_atr * (self.atr_period - 1) + current_tr) / self.atr_period

        # Calculate bands
        df['hl2'] = (df['high'] + df['low']) / 2
        df['basic_upperband'] = df['hl2'] + (self.factor * df['atr'])
        df['basic_lowerband'] = df['hl2'] - (self.factor * df['atr'])

        # Calculate final bands and SuperTrend
        df['final_upperband'] = 0.0
        df['final_lowerband'] = 0.0
        df['supertrend'] = 0.0
        df['direction'] = 1

        for i in range(len(df)):
            if i == 0:
                df.iloc[i, df.columns.get_loc('final_upperband')] = df.iloc[i]['basic_upperband']
                df.iloc[i, df.columns.get_loc('final_lowerband')] = df.iloc[i]['basic_lowerband']
                df.iloc[i, df.columns.get_loc('direction')] = 1
                df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['final_upperband']
            else:
                prev_final_upper = df.iloc[i-1]['final_upperband']
                prev_final_lower = df.iloc[i-1]['final_lowerband']
                prev_close = df.iloc[i-1]['close']
                prev_direction = df.iloc[i-1]['direction']
                
                current_basic_upper = df.iloc[i]['basic_upperband']
                current_basic_lower = df.iloc[i]['basic_lowerband']
                current_close = df.iloc[i]['close']

                if current_basic_upper < prev_final_upper or prev_close > prev_final_upper:
                    final_upper = current_basic_upper
                else:
                    final_upper = prev_final_upper

                if current_basic_lower > prev_final_lower or prev_close < prev_final_lower:
                    final_lower = current_basic_lower
                else:
                    final_lower = prev_final_lower

                df.iloc[i, df.columns.get_loc('final_upperband')] = final_upper
                df.iloc[i, df.columns.get_loc('final_lowerband')] = final_lower

                if prev_direction == 1:
                    if current_close > final_upper:
                        direction = -1
                    else:
                        direction = 1
                else:
                    if current_close < final_lower:
                        direction = 1
                    else:
                        direction = -1

                df.iloc[i, df.columns.get_loc('direction')] = direction

                if direction == -1:
                    supertrend_value = final_lower
                else:
                    supertrend_value = final_upper

                df.iloc[i, df.columns.get_loc('supertrend')] = supertrend_value

        return df
    
    def get_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Get trading signal from SuperTrend"""
        try:
            df_with_st = self.calculate_supertrend(df)
            
            latest_direction = df_with_st["direction"].iloc[-1]
            previous_direction = df_with_st["direction"].iloc[-2]
            
            signal_data = {
                "close": df_with_st["close"].iloc[-1],
                "supertrend": df_with_st["supertrend"].iloc[-1],
                "direction": latest_direction,
                "previous_direction": previous_direction,
                "trend": "GREEN (Uptrend)" if latest_direction == -1 else "RED (Downtrend)"
            }
            
            # Signal logic
            if previous_direction == 1 and latest_direction == -1:
                return "BUY", signal_data
            elif previous_direction == -1 and latest_direction == 1:
                return "SELL", signal_data
            else:
                return "HOLD", signal_data
                
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return "ERROR", {}
