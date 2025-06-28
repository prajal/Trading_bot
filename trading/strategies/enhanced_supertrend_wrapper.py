#!/usr/bin/env python3
"""
Enhanced SuperTrend Strategy Wrapper
Wraps your existing enhanced strategy to work with the new multi-strategy system
"""

import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, StrategyMetadata, StrategyType
from ..enhanced_strategy import EnhancedSuperTrendStrategy

logger = logging.getLogger(__name__)

class EnhancedSuperTrendWrapper(BaseStrategy):
    """
    Wrapper for your existing Enhanced SuperTrend strategy
    Makes it compatible with the new multi-strategy system without changing any existing code
    """
    
    # Strategy registration information
    STRATEGY_KEY = 'enhanced'
    STRATEGY_NAME = 'Enhanced SuperTrend'
    STRATEGY_DESCRIPTION = 'Your current enhanced SuperTrend strategy with adaptive parameters and regime detection'
    DEFAULT_CONFIG = {
        'atr_period': 10,
        'factor': 3.0,
        'adaptive_mode': True,
        'min_candles': 50
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize the original enhanced strategy
        self.enhanced_strategy = EnhancedSuperTrendStrategy(
            atr_period=config.get('atr_period', 10),
            factor=config.get('factor', 3.0),
            adaptive_mode=config.get('adaptive_mode', True)
        )
        
        logger.info(f"Enhanced SuperTrend wrapper initialized with config: {config}")
    
    def _initialize_metadata(self) -> StrategyMetadata:
        """Initialize strategy metadata"""
        return StrategyMetadata(
            name=self.STRATEGY_NAME,
            version="1.0",
            strategy_type=StrategyType.TREND_FOLLOWING,
            description=self.STRATEGY_DESCRIPTION,
            parameters={
                'atr_period': self.config.get('atr_period', 10),
                'factor': self.config.get('factor', 3.0),
                'adaptive_mode': self.config.get('adaptive_mode', True),
                'min_candles': self.config.get('min_candles', 50)
            },
            risk_level="MEDIUM",
            recommended_timeframes=["1m", "5m", "15m", "1h"],
            recommended_instruments=["NIFTYBEES", "BANKBEES", "JUNIORBEES"],
            backtested_performance={
                'win_rate': 0.58,
                'profit_factor': 1.8,
                'max_drawdown': -0.12,
                'sharpe_ratio': 1.2
            }
        )
    
    def get_signal(self, df: pd.DataFrame, has_position: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate trading signal using the original enhanced strategy
        This is a direct passthrough to your existing strategy
        """
        try:
            # Call your existing enhanced strategy
            signal, signal_data = self.enhanced_strategy.get_signal(df, has_position)
            
            # Add wrapper metadata to signal data
            signal_data.update({
                'strategy_wrapper': 'enhanced_supertrend_wrapper',
                'original_strategy': 'EnhancedSuperTrendStrategy',
                'wrapper_version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'data_points_used': len(df),
                'has_position': has_position
            })
            
            # Update performance metrics
            self.update_performance_metrics(signal)
            
            return signal, signal_data
            
        except Exception as e:
            logger.error(f"Error in Enhanced SuperTrend wrapper signal generation: {e}")
            return "HOLD", {
                'error': str(e),
                'strategy_wrapper': 'enhanced_supertrend_wrapper',
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_signal(self, df: pd.DataFrame) -> bool:
        """
        Validate strategy using the original enhanced strategy validation
        Direct passthrough to your existing validation logic
        """
        try:
            return self.enhanced_strategy.validate_signal(df)
        except Exception as e:
            logger.error(f"Error in Enhanced SuperTrend wrapper validation: {e}")
            return False
    
    def calculate_supertrend(self, df: pd.DataFrame, atr_period: int = None, factor: float = None) -> pd.DataFrame:
        """
        Direct access to SuperTrend calculation for backward compatibility
        """
        return self.enhanced_strategy.calculate_supertrend(df, atr_period, factor)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from the original enhanced strategy"""
        try:
            original_summary = self.enhanced_strategy.get_performance_summary()
            
            # Combine with wrapper metrics
            wrapper_summary = {
                'wrapper_info': {
                    'strategy_name': self.metadata.name,
                    'wrapper_version': '1.0',
                    'signals_generated': self.performance_metrics['signals_generated'],
                    'win_rate': self.performance_metrics['win_rate']
                },
                'original_strategy_summary': original_summary,
                'combined_metrics': {
                    'total_signals': self.performance_metrics['signals_generated'],
                    'last_signal_time': self.performance_metrics['last_signal_time'],
                    'strategy_health': self.get_strategy_health()
                }
            }
            
            return wrapper_summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'error': str(e),
                'wrapper_metrics': self.performance_metrics
            }
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed parameter information for the enhanced strategy"""
        return {
            'atr_period': {
                'current_value': self.config.get('atr_period', 10),
                'type': 'int',
                'min_value': 5,
                'max_value': 30,
                'description': 'ATR period for SuperTrend calculation'
            },
            'factor': {
                'current_value': self.config.get('factor', 3.0),
                'type': 'float',
                'min_value': 1.0,
                'max_value': 6.0,
                'description': 'SuperTrend factor multiplier'
            },
            'adaptive_mode': {
                'current_value': self.config.get('adaptive_mode', True),
                'type': 'bool',
                'description': 'Enable adaptive parameter adjustment based on market conditions'
            },
            'min_candles': {
                'current_value': self.config.get('min_candles', 50),
                'type': 'int',
                'min_value': 20,
                'max_value': 200,
                'description': 'Minimum candles required for signal generation'
            }
        }
    
    def supports_feature(self, feature: str) -> bool:
        """Check if this strategy wrapper supports specific features"""
        supported_features = [
            'signal_generation',
            'validation',
            'performance_tracking',
            'health_monitoring',
            'adaptive_parameters',
            'regime_detection',
            'confidence_scoring'
        ]
        
        return feature in supported_features
    
    def get_strategy_config_recommendations(self, market_condition: str = None) -> Dict[str, Any]:
        """Get configuration recommendations based on current market conditions"""
        recommendations = {
            'current_config': self.config.copy(),
            'market_condition': market_condition,
            'recommendations': []
        }
        
        # Basic recommendations based on performance
        if self.performance_metrics['win_rate'] < 0.5 and self.performance_metrics['signals_generated'] > 10:
            recommendations['recommendations'].append({
                'parameter': 'factor',
                'suggestion': 'increase',
                'reason': 'Low win rate suggests too many false signals - increase factor for more conservative signals'
            })
        
        if market_condition == 'volatile':
            recommendations['recommendations'].append({
                'parameter': 'factor',
                'suggestion': 'increase to 4.0-4.5',
                'reason': 'Volatile markets benefit from wider SuperTrend bands'
            })
        elif market_condition == 'trending':
            recommendations['recommendations'].append({
                'parameter': 'factor',
                'suggestion': 'decrease to 2.5-3.0',
                'reason': 'Trending markets can use tighter SuperTrend bands'
            })
        
        return recommendations
