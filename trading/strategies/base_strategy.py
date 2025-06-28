#!/usr/bin/env python3
"""
Base Strategy Interface for Multi-Strategy Trading System
Creates orthogonal strategy support without breaking existing code
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Strategy type classifications"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    HYBRID = "hybrid"

@dataclass
class StrategyMetadata:
    """Strategy metadata information"""
    name: str
    version: str
    strategy_type: StrategyType
    description: str
    parameters: Dict[str, Any]
    risk_level: str  # LOW, MEDIUM, HIGH
    recommended_timeframes: List[str]
    recommended_instruments: List[str]
    backtested_performance: Optional[Dict[str, float]] = None

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Ensures consistent interface across different strategy implementations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata = self._initialize_metadata()
        self.performance_metrics = {
            'signals_generated': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'last_signal_time': None
        }
        
        logger.info(f"Strategy initialized: {self.metadata.name} v{self.metadata.version}")
    
    @abstractmethod
    def _initialize_metadata(self) -> StrategyMetadata:
        """Initialize strategy metadata - must be implemented by each strategy"""
        pass
    
    @abstractmethod
    def get_signal(self, df: pd.DataFrame, has_position: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate trading signal
        
        Args:
            df: OHLCV data
            has_position: Whether currently holding a position
            
        Returns:
            Tuple of (signal, signal_data)
            signal: "BUY", "SELL", or "HOLD"
            signal_data: Dictionary with signal details
        """
        pass
    
    @abstractmethod
    def validate_signal(self, df: pd.DataFrame) -> bool:
        """
        Validate strategy calculations and data quality
        
        Args:
            df: OHLCV data to validate
            
        Returns:
            bool: True if validation passes
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        return {
            'metadata': {
                'name': self.metadata.name,
                'version': self.metadata.version,
                'type': self.metadata.strategy_type.value,
                'description': self.metadata.description,
                'risk_level': self.metadata.risk_level,
                'recommended_timeframes': self.metadata.recommended_timeframes,
                'recommended_instruments': self.metadata.recommended_instruments
            },
            'configuration': self.config,
            'performance': self.performance_metrics,
            'parameters': self.metadata.parameters
        }
    
    def get_required_data_length(self) -> int:
        """Get minimum data length required for strategy calculations"""
        # Default implementation - strategies can override
        return max(50, self.config.get('min_candles', 50))
    
    def update_performance_metrics(self, signal: str, was_successful: bool = None):
        """Update strategy performance tracking"""
        try:
            if signal in ["BUY", "SELL"]:
                self.performance_metrics['signals_generated'] += 1
                self.performance_metrics['last_signal_time'] = pd.Timestamp.now()
                
                if was_successful is not None and was_successful:
                    self.performance_metrics['successful_signals'] += 1
                
                # Update win rate
                if self.performance_metrics['signals_generated'] > 0:
                    self.performance_metrics['win_rate'] = (
                        self.performance_metrics['successful_signals'] / 
                        self.performance_metrics['signals_generated']
                    )
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_strategy_health(self) -> Dict[str, Any]:
        """Get strategy health assessment"""
        try:
            health = {
                'status': 'healthy',
                'issues': [],
                'recommendations': [],
                'metrics': self.performance_metrics.copy()
            }
            
            # Check signal generation rate
            if self.performance_metrics['signals_generated'] == 0:
                health['issues'].append("No signals generated yet")
            
            # Check win rate if we have enough signals
            if self.performance_metrics['signals_generated'] >= 10:
                win_rate = self.performance_metrics['win_rate']
                if win_rate < 0.4:
                    health['issues'].append(f"Low win rate: {win_rate:.1%}")
                    health['recommendations'].append("Consider adjusting strategy parameters")
                elif win_rate > 0.8:
                    health['recommendations'].append("Excellent performance - consider increasing position sizes")
            
            # Determine overall status
            if len(health['issues']) > 2:
                health['status'] = 'warning'
            elif len(health['issues']) > 5:
                health['status'] = 'critical'
            
            return health
            
        except Exception as e:
            logger.error(f"Error assessing strategy health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'metrics': self.performance_metrics.copy()
            }
    
    def supports_feature(self, feature: str) -> bool:
        """Check if strategy supports specific features"""
        # Base features all strategies should support
        base_features = [
            'signal_generation',
            'validation',
            'performance_tracking',
            'health_monitoring'
        ]
        
        return feature in base_features
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed parameter information for strategy tuning"""
        # Default implementation - strategies should override with actual parameters
        return {
            'min_candles': {
                'current_value': self.config.get('min_candles', 50),
                'type': 'int',
                'min_value': 10,
                'max_value': 200,
                'description': 'Minimum candles required for signal generation'
            }
        }
    
    def optimize_parameters(self, df: pd.DataFrame, 
                          optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Basic parameter optimization framework
        Strategies can override with specific optimization logic
        """
        logger.warning(f"Parameter optimization not implemented for {self.metadata.name}")
        return {
            'optimized': False,
            'reason': 'Optimization not implemented for this strategy',
            'current_parameters': self.metadata.parameters
        }
    
    def backtest_strategy(self, df: pd.DataFrame, 
                         initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Basic backtesting framework
        Strategies can override with specific backtesting logic
        """
        logger.warning(f"Backtesting not implemented for {self.metadata.name}")
        return {
            'backtest_completed': False,
            'reason': 'Backtesting not implemented for this strategy',
            'initial_capital': initial_capital
        }
    
    def __str__(self) -> str:
        """String representation of strategy"""
        return f"{self.metadata.name} v{self.metadata.version} ({self.metadata.strategy_type.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of strategy"""
        return (f"<{self.__class__.__name__}("
                f"name='{self.metadata.name}', "
                f"version='{self.metadata.version}', "
                f"type='{self.metadata.strategy_type.value}'"
                f")>")


class StrategyError(Exception):
    """Base exception for strategy-related errors"""
    pass

class StrategyValidationError(StrategyError):
    """Raised when strategy validation fails"""
    pass

class StrategyConfigurationError(StrategyError):
    """Raised when strategy configuration is invalid"""
    pass

class StrategyDataError(StrategyError):
    """Raised when strategy receives invalid data"""
    pass
