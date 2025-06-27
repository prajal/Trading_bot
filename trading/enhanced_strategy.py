import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedSuperTrendStrategy:
    """
    Enhanced SuperTrend strategy with adaptive parameters and signal confidence
    Drop-in replacement for your existing SuperTrendStrategy
    """
    
    def __init__(self, atr_period: int = 10, factor: float = 3.0, adaptive_mode: bool = True):
        self.base_atr_period = atr_period
        self.base_factor = factor
        self.adaptive_mode = adaptive_mode
        self.last_direction = None
        
        # Parameter sets for different market conditions
        self.parameter_sets = {
            "conservative": {"atr_period": 12, "factor": 3.5},
            "aggressive": {"atr_period": 8, "factor": 2.5}, 
            "volatile": {"atr_period": 7, "factor": 4.0},
            "default": {"atr_period": atr_period, "factor": factor}
        }
        
        # Current regime and parameters
        self.current_regime = "default"
        self.current_params = self.parameter_sets["default"]
        
        # Performance tracking
        self.signal_history = []
        
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime based on volatility and trend strength
        """
        if len(df) < 30:
            return "default"
        
        try:
            # Calculate recent volatility (last 20 periods)
            recent_returns = df['close'].pct_change().tail(20)
            current_volatility = recent_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate historical volatility baseline
            all_returns = df['close'].pct_change().dropna()
            avg_volatility = all_returns.std() * np.sqrt(252)
            
            # Calculate trend strength
            ma_20 = df['close'].rolling(20).mean()
            if len(ma_20.dropna()) > 0:
                trend_strength = abs((df['close'].iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1])
            else:
                trend_strength = 0
            
            # Regime classification logic
            vol_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
            
            if vol_ratio > 1.5:
                regime = "volatile"
            elif trend_strength > 0.05 and vol_ratio < 1.2:
                regime = "aggressive"  # Strong trend, normal volatility
            elif vol_ratio < 0.8:
                regime = "conservative"  # Low volatility
            else:
                regime = "default"
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "default"
    
    def get_adaptive_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get parameters based on current market regime"""
        if not self.adaptive_mode:
            return self.parameter_sets["default"]
        
        regime = self.detect_market_regime(df)
        
        if regime != self.current_regime:
            logger.info(f"📊 Market regime changed: {self.current_regime} → {regime}")
            self.current_regime = regime
        
        return self.parameter_sets[regime]
    
    def rma(self, series, length):
        """Wilder's RMA (TradingView's ta.rma)"""
        alpha = 1 / length
        rma = series.copy()
        rma.iloc[0] = series.iloc[:length].mean()
        for i in range(1, len(series)):
            rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i-1]
        return rma

    def calculate_supertrend(self, df: pd.DataFrame, atr_period: int = None, factor: float = None) -> pd.DataFrame:
        """
        SuperTrend implementation matching TradingView's ta.supertrend.
        Returns DataFrame with columns: ['supertrend', 'direction', 'trend_desc']
        """
        if atr_period is None:
            atr_period = self.base_atr_period
        if factor is None:
            factor = self.base_factor
        df = df.copy()
        hl2 = (df['high'] + df['low']) / 2
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = self.rma(tr, atr_period)
        upperband = hl2 + (factor * atr)
        lowerband = hl2 - (factor * atr)
        df['atr'] = atr

        direction = pd.Series(index=df.index, dtype=int)
        supertrend = pd.Series(index=df.index, dtype=float)
        direction.iloc[0] = 1
        supertrend.iloc[0] = lowerband.iloc[0]

        for i in range(1, len(df)):
            prev_st = supertrend.iloc[i-1]
            prev_dir = direction.iloc[i-1]
            curr_close = df['close'].iloc[i]

            # Band logic
            if upperband.iloc[i] < upperband.iloc[i-1] or df['close'].iloc[i-1] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i]
            else:
                upperband.iloc[i] = upperband.iloc[i-1]
            if lowerband.iloc[i] > lowerband.iloc[i-1] or df['close'].iloc[i-1] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i]
            else:
                lowerband.iloc[i] = lowerband.iloc[i-1]

            # Direction logic
            if prev_dir == -1 and curr_close > upperband.iloc[i-1]:
                direction.iloc[i] = 1
            elif prev_dir == 1 and curr_close < lowerband.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = prev_dir

            # SuperTrend value
            supertrend.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]

        df['supertrend'] = supertrend
        df['direction'] = direction
        df['trend_desc'] = df['direction'].map({1: 'Uptrend', -1: 'Downtrend'})
        return df
    
    def calculate_signal_confidence(self, df: pd.DataFrame) -> float:
        """Calculate signal confidence based on multiple factors"""
        if len(df) < 20:
            return 0.5  # Neutral confidence
        
        try:
            factors = []
            
            # 1. ATR Stability (less volatile ATR = higher confidence)
            recent_atr = df['atr'].tail(10)
            if len(recent_atr) > 1 and recent_atr.mean() > 0:
                atr_stability = max(0, 1.0 - (recent_atr.std() / recent_atr.mean()))
                factors.append(atr_stability * 0.25)
            
            # 2. Trend Strength (distance from SuperTrend)
            current_close = df['close'].iloc[-1]
            current_supertrend = df['supertrend'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            if current_atr > 0:
                price_st_distance = abs(current_close - current_supertrend) / current_atr
                trend_strength = min(price_st_distance / 2.0, 1.0)
                factors.append(trend_strength * 0.30)
            
            # 3. Direction Persistence
            direction_changes = (df['direction'].diff() != 0).sum()
            if len(df) > 0:
                direction_persistence = max(0, 1.0 - (direction_changes / len(df)))
                factors.append(direction_persistence * 0.25)
            
            # 4. Volume Confirmation (if available)
            if 'volume' in df.columns:
                recent_volume = df['volume'].tail(5).mean()
                historical_volume = df['volume'].tail(20).mean()
                if historical_volume > 0:
                    volume_ratio = min(recent_volume / historical_volume, 2.0) / 2.0
                    factors.append(volume_ratio * 0.20)
                else:
                    factors.append(0.6)  # Neutral
            else:
                factors.append(0.6)  # Neutral when no volume data
            
            total_confidence = sum(factors)
            return max(0.1, min(1.0, total_confidence))  # Clamp between 0.1 and 1.0
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    def get_signal(self, df: pd.DataFrame, has_position: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Get trading signal with enhanced logic and confidence scoring"""
        try:
            df_with_st = self.calculate_supertrend(df)
            
            if len(df_with_st) < 2:
                return "HOLD", {"error": "Insufficient data for signal generation"}
            
            # Current and previous values
            current_direction = df_with_st["direction"].iloc[-1]
            previous_direction = df_with_st["direction"].iloc[-2]
            current_close = df_with_st["close"].iloc[-1]
            current_supertrend = df_with_st["supertrend"].iloc[-1]
            
            # Calculate signal confidence
            confidence = self.calculate_signal_confidence(df_with_st)
            
            # Create signal data
            signal_data = {
                "close": current_close,
                "supertrend": current_supertrend,
                "direction": current_direction,
                "previous_direction": previous_direction,
                "regime": self.current_regime,
                "atr_period": self.current_params["atr_period"],
                "factor": self.current_params["factor"],
                "confidence": confidence,
                "trend": "GREEN (Uptrend)" if current_direction == 1 else "RED (Downtrend)",
                "price_vs_supertrend": "Above" if current_close > current_supertrend else "Below",
                "atr": df_with_st["atr"].iloc[-1]
            }
            
            # Direction change detection with confidence threshold
            direction_changed = self.last_direction is not None and self.last_direction != current_direction
            min_confidence = 0.6  # Require 60% confidence for trades
            
            # BUY Signal: RED to GREEN transition
            if (direction_changed and self.last_direction == -1 and current_direction == 1 and 
                not has_position and confidence >= min_confidence):
                
                logger.info(f"🟢 ENHANCED BUY SIGNAL:")
                logger.info(f"   Direction: {self.last_direction} → {current_direction} (RED→GREEN)")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Regime: {self.current_regime}")
                logger.info(f"   Parameters: ATR={self.current_params['atr_period']}, Factor={self.current_params['factor']}")
                
                self.last_direction = current_direction
                return "BUY", signal_data
            
            # Alternative BUY detection (fallback)
            elif (previous_direction == -1 and current_direction == 1 and 
                  not has_position and confidence >= min_confidence):
                
                logger.info(f"🟢 ENHANCED BUY SIGNAL (Alt detection):")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Regime: {self.current_regime}")
                
                self.last_direction = current_direction
                return "BUY", signal_data
            
            # SELL Signal: GREEN to RED transition
            elif (direction_changed and self.last_direction == 1 and current_direction == -1 and 
                  has_position and confidence >= min_confidence):
                
                logger.info(f"🔴 ENHANCED SELL SIGNAL:")
                logger.info(f"   Direction: {self.last_direction} → {current_direction} (GREEN→RED)")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Regime: {self.current_regime}")
                
                self.last_direction = current_direction
                return "SELL", signal_data
            
            # Alternative SELL detection (fallback)
            elif (previous_direction == 1 and current_direction == -1 and 
                  has_position and confidence >= min_confidence):
                
                logger.info(f"🔴 ENHANCED SELL SIGNAL (Alt detection):")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Regime: {self.current_regime}")
                
                self.last_direction = current_direction
                return "SELL", signal_data
            
            # Low confidence filter
            elif confidence < min_confidence:
                logger.debug(f"🚫 Signal filtered due to low confidence: {confidence:.2f}")
                signal_data["reason"] = f"Low confidence ({confidence:.2f} < {min_confidence})"
                self.last_direction = current_direction
                return "HOLD", signal_data
            
            # No signal
            else:
                self.last_direction = current_direction
                logger.debug(f"📊 Market Status: {signal_data['trend']} | Confidence: {confidence:.2f} | Regime: {self.current_regime}")
                return "HOLD", signal_data
                
        except Exception as e:
            logger.error(f"Error calculating enhanced SuperTrend signal: {e}")
            return "ERROR", {"error": str(e)}
    
    def validate_signal(self, df: pd.DataFrame) -> bool:
        """Validate that SuperTrend is working correctly"""
        try:
            df_with_st = self.calculate_supertrend(df)
            
            # Check for NaN values
            if df_with_st['atr'].isna().any():
                logger.error("ATR contains NaN values")
                return False
            
            if df_with_st['supertrend'].isna().any():
                logger.error("SuperTrend contains NaN values")
                return False
            
            # Check direction values are valid
            valid_directions = df_with_st['direction'].isin([1, -1]).all()
            if not valid_directions:
                logger.error("Invalid direction values found")
                return False
            
            # Count direction changes
            direction_changes = (df_with_st['direction'].diff() != 0).sum() - 1
            logger.info(f"✅ Enhanced SuperTrend validation passed")
            logger.info(f"   Direction changes: {direction_changes}")
            logger.info(f"   Current regime: {self.current_regime}")
            logger.info(f"   Parameters: ATR={self.current_params['atr_period']}, Factor={self.current_params['factor']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced SuperTrend validation failed: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the enhanced strategy"""
        return {
            "current_regime": self.current_regime,
            "current_parameters": self.current_params,
            "adaptive_mode": self.adaptive_mode,
            "signal_count": len(self.signal_history),
            "parameter_sets": self.parameter_sets
        }

# Backward compatibility - alias for your existing code
SuperTrendStrategy = EnhancedSuperTrendStrategy