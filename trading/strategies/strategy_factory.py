#!/usr/bin/env python3
"""
Strategy Factory and Manager for Multi-Strategy Trading System
Handles strategy creation, selection, and management
"""

from typing import Dict, Any, List, Type, Optional
import logging
from pathlib import Path
import importlib
import inspect

from .base_strategy import BaseStrategy, StrategyType, StrategyError

logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory for creating and managing trading strategy instances
    Supports dynamic strategy loading and registration
    """
    
    # Registry of available strategies
    _strategies: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_strategy(cls, 
                         strategy_key: str, 
                         strategy_class: Type[BaseStrategy],
                         name: str,
                         description: str,
                         default_config: Dict[str, Any] = None):
        """Register a new strategy"""
        if not issubclass(strategy_class, BaseStrategy):
            raise StrategyError(f"Strategy class must inherit from BaseStrategy")
        
        cls._strategies[strategy_key] = {
            'class': strategy_class,
            'name': name,
            'description': description,
            'default_config': default_config or {}
        }
        
        logger.info(f"Registered strategy: {strategy_key} - {name}")
    
    @classmethod
    def create_strategy(cls, strategy_key: str, config: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance"""
        if strategy_key not in cls._strategies:
            available = list(cls._strategies.keys())
            raise StrategyError(f"Unknown strategy '{strategy_key}'. Available: {available}")
        
        strategy_info = cls._strategies[strategy_key]
        strategy_class = strategy_info['class']
        
        # Merge default config with provided config
        final_config = strategy_info['default_config'].copy()
        if config:
            final_config.update(config)
        
        try:
            instance = strategy_class(final_config)
            logger.info(f"Created strategy instance: {strategy_key}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create strategy '{strategy_key}': {e}")
            raise StrategyError(f"Strategy creation failed: {e}")
    
    @classmethod
    def list_strategies(cls) -> Dict[str, Dict[str, str]]:
        """List all available strategies"""
        return {
            key: {
                'name': info['name'],
                'description': info['description'],
                'class_name': info['class'].__name__
            }
            for key, info in cls._strategies.items()
        }
    
    @classmethod
    def get_strategy_info(cls, strategy_key: str) -> Dict[str, Any]:
        """Get detailed information about a strategy"""
        if strategy_key not in cls._strategies:
            raise StrategyError(f"Unknown strategy: {strategy_key}")
        
        strategy_info = cls._strategies[strategy_key]
        
        # Create temporary instance to get metadata
        try:
            temp_instance = cls.create_strategy(strategy_key)
            metadata = temp_instance.get_strategy_info()
            
            return {
                'key': strategy_key,
                'factory_info': {
                    'name': strategy_info['name'],
                    'description': strategy_info['description'],
                    'default_config': strategy_info['default_config']
                },
                'strategy_metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error getting strategy info for '{strategy_key}': {e}")
            return {
                'key': strategy_key,
                'factory_info': {
                    'name': strategy_info['name'],
                    'description': strategy_info['description'],
                    'default_config': strategy_info['default_config']
                },
                'error': str(e)
            }
    
    @classmethod
    def auto_discover_strategies(cls, strategies_dir: Path):
        """Automatically discover and register strategies from directory"""
        try:
            if not strategies_dir.exists():
                logger.warning(f"Strategies directory not found: {strategies_dir}")
                return
            
            for strategy_file in strategies_dir.glob("*_strategy.py"):
                if strategy_file.name.startswith("__"):
                    continue
                
                try:
                    # Import the module
                    module_name = strategy_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find strategy classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != BaseStrategy and 
                            issubclass(obj, BaseStrategy) and 
                            hasattr(obj, 'STRATEGY_KEY')):
                            
                            strategy_key = obj.STRATEGY_KEY
                            strategy_name = getattr(obj, 'STRATEGY_NAME', name)
                            strategy_desc = getattr(obj, 'STRATEGY_DESCRIPTION', f"Auto-discovered: {name}")
                            default_config = getattr(obj, 'DEFAULT_CONFIG', {})
                            
                            cls.register_strategy(
                                strategy_key=strategy_key,
                                strategy_class=obj,
                                name=strategy_name,
                                description=strategy_desc,
                                default_config=default_config
                            )
                            
                except Exception as e:
                    logger.error(f"Error discovering strategy in {strategy_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during strategy auto-discovery: {e}")


class StrategyManager:
    """
    Manager for handling multiple strategies and strategy switching
    """
    
    def __init__(self):
        self.current_strategy: Optional[BaseStrategy] = None
        self.strategy_history: List[Dict[str, Any]] = []
        self.performance_tracker = {}
    
    def set_strategy(self, strategy_key: str, config: Dict[str, Any] = None) -> BaseStrategy:
        """Set the current active strategy"""
        try:
            # Create new strategy instance
            new_strategy = StrategyFactory.create_strategy(strategy_key, config)
            
            # Record strategy change
            change_record = {
                'timestamp': pd.Timestamp.now(),
                'from_strategy': self.current_strategy.metadata.name if self.current_strategy else None,
                'to_strategy': new_strategy.metadata.name,
                'config': config or {}
            }
            self.strategy_history.append(change_record)
            
            # Update current strategy
            self.current_strategy = new_strategy
            
            logger.info(f"Strategy changed to: {new_strategy}")
            return new_strategy
            
        except Exception as e:
            logger.error(f"Failed to set strategy '{strategy_key}': {e}")
            raise
    
    def get_current_strategy(self) -> Optional[BaseStrategy]:
        """Get the currently active strategy"""
        return self.current_strategy
    
    def validate_current_strategy(self, df) -> bool:
        """Validate the current strategy"""
        if not self.current_strategy:
            logger.error("No strategy is currently set")
            return False
        
        try:
            return self.current_strategy.validate_signal(df)
        except Exception as e:
            logger.error(f"Strategy validation failed: {e}")
            return False
    
    def get_signal(self, df, has_position: bool = False):
        """Get signal from current strategy"""
        if not self.current_strategy:
            raise StrategyError("No strategy is currently set")
        
        try:
            signal, signal_data = self.current_strategy.get_signal(df, has_position)
            
            # Update performance tracking
            self.current_strategy.update_performance_metrics(signal)
            
            # Add strategy identification to signal data
            signal_data['strategy_info'] = {
                'name': self.current_strategy.metadata.name,
                'version': self.current_strategy.metadata.version,
                'type': self.current_strategy.metadata.strategy_type.value
            }
            
            return signal, signal_data
            
        except Exception as e:
            logger.error(f"Error getting signal from strategy: {e}")
            raise StrategyError(f"Signal generation failed: {e}")
    
    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all used strategies"""
        summary = {
            'current_strategy': None,
            'strategy_changes': len(self.strategy_history),
            'strategies_used': [],
            'performance_by_strategy': {}
        }
        
        if self.current_strategy:
            summary['current_strategy'] = {
                'name': self.current_strategy.metadata.name,
                'version': self.current_strategy.metadata.version,
                'health': self.current_strategy.get_strategy_health()
            }
        
        # Analyze strategy history
        strategy_names = set()
        for record in self.strategy_history:
            if record['to_strategy']:
                strategy_names.add(record['to_strategy'])
        
        summary['strategies_used'] = list(strategy_names)
        
        return summary
    
    def compare_strategies(self, strategy_keys: List[str], 
                          test_data, comparison_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compare multiple strategies on test data"""
        results = {}
        
        for strategy_key in strategy_keys:
            try:
                # Create strategy instance
                strategy = StrategyFactory.create_strategy(strategy_key, comparison_config)
                
                # Test strategy
                if strategy.validate_signal(test_data):
                    signal, signal_data = strategy.get_signal(test_data)
                    
                    results[strategy_key] = {
                        'strategy_name': strategy.metadata.name,
                        'signal': signal,
                        'signal_data': signal_data,
                        'health': strategy.get_strategy_health(),
                        'info': strategy.get_strategy_info()
                    }
                else:
                    results[strategy_key] = {
                        'strategy_name': strategy.metadata.name,
                        'error': 'Strategy validation failed'
                    }
                    
            except Exception as e:
                results[strategy_key] = {
                    'error': f"Strategy test failed: {e}"
                }
        
        return results


# Initialize the factory with built-in strategies
def initialize_builtin_strategies():
    """Initialize built-in strategies"""
    try:
        # Register Enhanced SuperTrend (wrapper for existing strategy)
        from .enhanced_supertrend_wrapper import EnhancedSuperTrendWrapper
        StrategyFactory.register_strategy(
            strategy_key='enhanced',
            strategy_class=EnhancedSuperTrendWrapper,
            name='Enhanced SuperTrend',
            description='Your current enhanced SuperTrend strategy with adaptive parameters',
            default_config={
                'atr_period': 10,
                'factor': 3.0,
                'adaptive_mode': True,
                'min_candles': 50
            }
        )
        
        # Register Bulletproof SuperTrend
        from .bulletproof_supertrend_strategy import BulletproofSuperTrendStrategy
        StrategyFactory.register_strategy(
            strategy_key='bullet',
            strategy_class=BulletproofSuperTrendStrategy,
            name='Bulletproof SuperTrend',
            description='Rock-solid SuperTrend with advanced quality filtering and risk management',
            default_config={
                'base_atr_period': 10,
                'base_factor': 3.0,
                'adaptive_mode': True,
                'quality_threshold': 0.65,
                'min_candles': 50
            }
        )
        
        logger.info("Built-in strategies initialized successfully")
        
    except ImportError as e:
        logger.warning(f"Some built-in strategies not available: {e}")
    except Exception as e:
        logger.error(f"Error initializing built-in strategies: {e}")


# Auto-initialize when module is imported
try:
    initialize_builtin_strategies()
except Exception as e:
    logger.error(f"Failed to initialize built-in strategies: {e}")


# Create global strategy manager instance
strategy_manager = StrategyManager()
