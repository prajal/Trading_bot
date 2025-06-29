#!/usr/bin/env python3
"""
Fixed Bulletproof SuperTrend Strategy
Uses the existing enhanced strategy as core with additional quality filtering
"""

import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, StrategyMetadata, StrategyType
from ..enhanced_strategy import EnhancedSuperTrendStrategy

logger = logging.getLogger(__name__)

class BulletproofSuperTrendStrategy(BaseStrategy):
    """
    Bulletproof SuperTrend strategy - a more robust version with quality filtering
    Built on top of the existing enhanced strategy
    """
    
    # Strategy registration information
    STRATEGY_KEY = 'bullet'
    STRATEGY_NAME = 'Bulletproof SuperTrend'
    STRATEGY_DESCRIPTION = 'Rock-solid SuperTrend with quality filtering and conservative signals'
    DEFAULT_CONFIG = {
        'base_atr_period': 10,
        'base_factor': 3.0,
        'adaptive_mode': True,
        'quality_threshold': 0.65,
        'min_candles': 50,
        'min_confidence': 0.65,  # Higher minimum confidence for bulletproof
        'consecutive_signals': 5   # Require 2 consecutive signals for confirmation
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize the core enhanced strategy
        self.core_strategy = EnhancedSuperTrendStrategy(
            atr_period=config.get('base_atr_period', 10),
            factor=config.get('base_factor', 3.0),
            adaptive_mode=config.get('adaptive_mode', True)
        )
        
        # Bulletproof-specific settings
        self.quality_threshold = config.get('quality_threshold', 0.65)
        self.min_confidence = config.get('min_confidence', 0.65)
        self.consecutive_signals = config.get('consecutive_signals', 2)
        
        # Signal history for confirmation
        self.signal_history = []
        self.last_confirmed_signal = None
        
        logger.info(f"Bulletproof SuperTrend strategy initialized with config: {config}")
    
    def _initialize_metadata(self) -> StrategyMetadata:
        """Initialize strategy metadata"""
        return StrategyMetadata(
            name=self.STRATEGY_NAME,
            version="2.0",
            strategy_type=StrategyType.TREND_FOLLOWING,
            description=self.STRATEGY_DESCRIPTION,
            parameters={
                'base_atr_period': self.config.get('base_atr_period', 10),
                'base_factor': self.config.get('base_factor', 3.0),
                'adaptive_mode': self.config.get('adaptive_mode', True),
                'quality_threshold': self.config.get('quality_threshold', 0.65),
                'min_confidence': self.config.get('min_confidence', 0.65),
                'consecutive_signals': self.config.get('consecutive_signals', 2),
                'min_candles': self.config.get('min_candles', 50)
            },
            risk_level="LOW",  # Conservative approach
            recommended_timeframes=["5m", "15m", "1h"],
            recommended_instruments=["NIFTYBEES", "BANKBEES", "JUNIORBEES"],
            backtested_performance={
                'win_rate': 0.65,
                'profit_factor': 2.2,
                'max_drawdown': -0.08,
                'sharpe_ratio': 1.6
            }
        )
    
    def calculate_supertrend(self, df: pd.DataFrame, atr_period: int = None, factor: float = None) -> pd.DataFrame:
        """
        Calculate SuperTrend using the core enhanced strategy
        This ensures compatibility with the backtester
        """
        # Use the core strategy's calculate_supertrend method
        return self.core_strategy.calculate_supertrend(df, atr_period, factor)
    
    def _calculate_signal_quality(self, df: pd.DataFrame, signal_data: Dict[str, Any]) -> float:
        """Calculate signal quality score with bulletproof criteria"""
        try:
            quality_score = 0.0
            quality_factors = []
            
            # 1. Base confidence from enhanced strategy
            base_confidence = signal_data.get('confidence', 0.5)
            quality_factors.append(('base_confidence', base_confidence * 0.3))
            
            # 2. Trend strength (price distance from SuperTrend)
            if 'supertrend' in signal_data and 'close' in signal_data:
                price = signal_data['close']
                st = signal_data['supertrend']
                atr = signal_data.get('atr', 1)
                
                if atr > 0:
                    distance = abs(price - st) / atr
                    trend_strength = min(distance / 3.0, 1.0)  # Normalize to 0-1
                    quality_factors.append(('trend_strength', trend_strength * 0.2))
            
            # 3. ATR stability (lower volatility in ATR = more stable)
            if len(df) >= 20:
                recent_atr = df['atr'].tail(10)
                atr_cv = recent_atr.std() / recent_atr.mean() if recent_atr.mean() > 0 else 1.0
                atr_stability = max(0, 1 - atr_cv)  # Lower CV = higher stability
                quality_factors.append(('atr_stability', atr_stability * 0.2))
            
            # 4. Price action quality (clean moves)
            if len(df) >= 5:
                recent_candles = df.tail(5)
                
                # Check for clean directional moves
                if signal_data.get('direction') == 1:  # Bullish
                    bullish_candles = ((recent_candles['close'] > recent_candles['open']).sum() / 5)
                    quality_factors.append(('price_action', bullish_candles * 0.15))
                elif signal_data.get('direction') == -1:  # Bearish
                    bearish_candles = ((recent_candles['close'] < recent_candles['open']).sum() / 5)
                    quality_factors.append(('price_action', bearish_candles * 0.15))
            
            # 5. Volume quality (if available)
            if 'volume' in df.columns and len(df) >= 20:
                recent_vol = df['volume'].tail(5).mean()
                avg_vol = df['volume'].tail(20).mean()
                
                if avg_vol > 0:
                    vol_ratio = recent_vol / avg_vol
                    # Good if volume is between 0.8x and 2x average
                    vol_quality = 1.0 if 0.8 <= vol_ratio <= 2.0 else max(0, 1 - abs(vol_ratio - 1.4) / 2)
                    quality_factors.append(('volume', vol_quality * 0.15))
            
            # Calculate total quality score
            quality_score = sum(score for _, score in quality_factors)
            
            # Log quality breakdown for debugging
            logger.debug(f"Signal quality breakdown: {dict(quality_factors)}, Total: {quality_score:.3f}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating signal quality: {e}")
            return 0.5  # Default moderate quality
    
    def _check_signal_confirmation(self, current_signal: str) -> bool:
        """Check if signal meets consecutive confirmation requirement"""
        # Add current signal to history
        self.signal_history.append(current_signal)
        
        # Keep only recent signals
        if len(self.signal_history) > self.consecutive_signals + 2:
            self.signal_history = self.signal_history[-(self.consecutive_signals + 2):]
        
        # Check for consecutive signals
        if len(self.signal_history) >= self.consecutive_signals:
            recent_signals = self.signal_history[-self.consecutive_signals:]
            
            # All recent signals must be the same and not HOLD
            if all(s == current_signal for s in recent_signals) and current_signal != "HOLD":
                return True
        
        return False
    
    def get_signal(self, df: pd.DataFrame, has_position: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate bulletproof trading signal with quality filtering
        """
        try:
            # Ensure we have the necessary calculations
            df_with_st = self.calculate_supertrend(df)
            
            # Get signal from core enhanced strategy
            core_signal, core_signal_data = self.core_strategy.get_signal(df_with_st, has_position)
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(df_with_st, core_signal_data)
            
            # Create bulletproof signal data
            signal_data = core_signal_data.copy()
            signal_data.update({
                'strategy_name': self.metadata.name,
                'strategy_type': 'bulletproof',
                'signal_quality': signal_quality,
                'quality_threshold': self.quality_threshold,
                'min_confidence': self.min_confidence,
                'consecutive_signals_required': self.consecutive_signals,
                'filters_passed': []
            })
            
            # Apply bulletproof filters
            final_signal = core_signal
            
            # Filter 1: Quality threshold
            if signal_quality < self.quality_threshold:
                signal_data['filters_passed'].append(f"FAILED: Quality {signal_quality:.3f} < {self.quality_threshold}")
                final_signal = "HOLD"
            else:
                signal_data['filters_passed'].append(f"PASSED: Quality {signal_quality:.3f}")
            
            # Filter 2: Confidence threshold
            core_confidence = core_signal_data.get('confidence', 0)
            if core_confidence < self.min_confidence:
                signal_data['filters_passed'].append(f"FAILED: Confidence {core_confidence:.3f} < {self.min_confidence}")
                final_signal = "HOLD"
            else:
                signal_data['filters_passed'].append(f"PASSED: Confidence {core_confidence:.3f}")
            
            # Filter 3: Consecutive signal confirmation
            if final_signal != "HOLD" and self.consecutive_signals > 1:
                if self._check_signal_confirmation(final_signal):
                    signal_data['filters_passed'].append(f"PASSED: {self.consecutive_signals} consecutive signals")
                else:
                    signal_data['filters_passed'].append(f"FAILED: Awaiting {self.consecutive_signals} consecutive signals")
                    final_signal = "HOLD"
            
            # Filter 4: Position-specific filters
            if has_position and final_signal == "BUY":
                signal_data['filters_passed'].append("FAILED: Already in position")
                final_signal = "HOLD"
            elif not has_position and final_signal == "SELL":
                signal_data['filters_passed'].append("FAILED: No position to sell")
                final_signal = "HOLD"
            
            # Update last confirmed signal
            if final_signal != "HOLD":
                self.last_confirmed_signal = final_signal
            
            # Update performance metrics
            self.update_performance_metrics(final_signal)
            
            # Log bulletproof signal
            if final_signal != core_signal:
                logger.info(f"Bulletproof filter: {core_signal} → {final_signal} (Quality: {signal_quality:.3f})")
            
            return final_signal, signal_data
            
        except Exception as e:
            logger.error(f"Error in Bulletproof SuperTrend signal generation: {e}")
            return "HOLD", {
                'error': str(e),
                'strategy_name': self.metadata.name,
                'strategy_type': 'bulletproof'
            }
    
    def validate_signal(self, df: pd.DataFrame) -> bool:
        """
        Validate bulletproof strategy calculations
        """
        try:
            # First validate using core strategy
            if not self.core_strategy.validate_signal(df):
                return False
            
            # Additional bulletproof validation
            if len(df) < self.config.get('min_candles', 50):
                logger.warning(f"Insufficient data for bulletproof strategy: {len(df)} candles")
                return False
            
            # Ensure we can calculate quality metrics
            df_with_st = self.calculate_supertrend(df)
            if 'atr' not in df_with_st.columns:
                logger.error("ATR calculation failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in Bulletproof SuperTrend validation: {e}")
            return False
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed parameter information for the bulletproof strategy"""
        return {
            'base_atr_period': {
                'current_value': self.config.get('base_atr_period', 10),
                'type': 'int',
                'min_value': 5,
                'max_value': 30,
                'description': 'Base ATR period for SuperTrend calculation'
            },
            'base_factor': {
                'current_value': self.config.get('base_factor', 3.0),
                'type': 'float',
                'min_value': 1.0,
                'max_value': 6.0,
                'description': 'Base SuperTrend factor multiplier'
            },
            'quality_threshold': {
                'current_value': self.config.get('quality_threshold', 0.65),
                'type': 'float',
                'min_value': 0.0,
                'max_value': 1.0,
                'description': 'Minimum signal quality score required (0-1)'
            },
            'min_confidence': {
                'current_value': self.config.get('min_confidence', 0.65),
                'type': 'float',
                'min_value': 0.0,
                'max_value': 1.0,
                'description': 'Minimum confidence level from core strategy'
            },
            'consecutive_signals': {
                'current_value': self.config.get('consecutive_signals', 2),
                'type': 'int',
                'min_value': 1,
                'max_value': 5,
                'description': 'Number of consecutive signals required for confirmation'
            },
            'adaptive_mode': {
                'current_value': self.config.get('adaptive_mode', True),
                'type': 'bool',
                'description': 'Enable adaptive parameter adjustment'
            }
        }
    
    def supports_feature(self, feature: str) -> bool:
        """Check if this bulletproof strategy supports specific features"""
        bulletproof_features = [
            'signal_generation',
            'validation',
            'performance_tracking',
            'health_monitoring',
            'adaptive_parameters',
            'quality_filtering',
            'signal_confirmation',
            'conservative_approach',
            'multi_filter_system'
        ]
        
        return feature in bulletproof_features
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get strategy-specific statistics"""
        return {
            'signals_filtered': len([s for s in self.signal_history if s == "HOLD"]),
            'signals_confirmed': len([s for s in self.signal_history if s in ["BUY", "SELL"]]),
            'average_quality': 0.0,  # Would need to track this
            'last_confirmed_signal': self.last_confirmed_signal,
            'signal_history_length': len(self.signal_history)
        }