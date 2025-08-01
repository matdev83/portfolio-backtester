"""
Configuration validation utilities for timing framework.
"""

from typing import Dict, List, Any


class TimingConfigValidator:
    """Validates timing configuration parameters."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Validate timing configuration and return list of errors."""
        errors = []
        
        # Validate mode
        mode = config.get('mode', 'time_based')
        valid_modes = ['time_based', 'signal_based']
        if mode not in valid_modes:
            errors.append(f"Invalid timing mode '{mode}'. Must be one of {valid_modes}")
            return errors  # Return early if mode is invalid
        
        # Validate based on mode
        if mode == 'time_based':
            errors.extend(TimingConfigValidator.validate_time_based_config(config))
        elif mode == 'signal_based':
            errors.extend(TimingConfigValidator.validate_signal_based_config(config))
        
        return errors
    
    @staticmethod
    def validate_time_based_config(config: Dict[str, Any]) -> List[str]:
        """Validate time-based timing configuration."""
        errors = []
        
        frequency = config.get('rebalance_frequency', 'M')
        # Comprehensive list of pandas frequencies for rebalancing
        valid_frequencies = [
            # Daily and weekly
            'D', 'B', 'W', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
            # Monthly
            'M', 'ME', 'BM', 'BMS', 'MS',
            # Quarterly  
            'Q', 'QE', 'QS', 'BQ', 'BQS', '2Q',
            # Semi-annual
            '6M', '6ME', '6MS',
            # Annual
            'A', 'AS', 'Y', 'YE', 'YS', 'BA', 'BAS', 'BY', 'BYS', '2A',
            # Hourly (for high-frequency strategies)
            'H', '2H', '3H', '4H', '6H', '8H', '12H'
        ]
        if frequency not in valid_frequencies:
            errors.append(f"Invalid rebalance_frequency '{frequency}'. Must be one of {valid_frequencies}. "
                         f"Common values: 'M' (monthly), 'ME' (month-end), 'Q' (quarterly), 'QE' (quarter-end), "
                         f"'6M' (semi-annual), 'A' (annual), 'YE' (year-end), 'D' (daily), 'W' (weekly)")
        
        offset = config.get('rebalance_offset', 0)
        if not isinstance(offset, int) or abs(offset) > 30:
            errors.append(f"rebalance_offset must be an integer between -30 and 30, got {offset}")
        
        return errors
    
    @staticmethod
    def validate_signal_based_config(config: Dict[str, Any]) -> List[str]:
        """Validate signal-based timing configuration."""
        errors = []
        
        scan_freq = config.get('scan_frequency', 'D')
        valid_scan_frequencies = ['D', 'W', 'M']
        if scan_freq not in valid_scan_frequencies:
            errors.append(f"Invalid scan_frequency '{scan_freq}'. Must be one of {valid_scan_frequencies}")
        
        max_holding = config.get('max_holding_period')
        max_holding_valid = True
        if max_holding is not None and (not isinstance(max_holding, int) or max_holding < 1):
            errors.append(f"max_holding_period must be a positive integer, got {max_holding}")
            max_holding_valid = False
        
        min_holding = config.get('min_holding_period', 1)
        min_holding_valid = True
        if not isinstance(min_holding, int) or min_holding < 1:
            errors.append(f"min_holding_period must be a positive integer, got {min_holding}")
            min_holding_valid = False
        
        # Only check min vs max if both are valid integers and positive
        if (max_holding is not None and max_holding_valid and min_holding_valid and 
            min_holding > max_holding):
            errors.append(f"min_holding_period ({min_holding}) cannot exceed max_holding_period ({max_holding})")
        
        return errors
    
    @staticmethod
    def get_default_config(mode: str = 'time_based') -> Dict[str, Any]:
        """Get default configuration for a timing mode."""
        if mode == 'time_based':
            return {
                'mode': 'time_based',
                'rebalance_frequency': 'M',
                'rebalance_offset': 0
            }
        elif mode == 'signal_based':
            return {
                'mode': 'signal_based',
                'scan_frequency': 'D',
                'min_holding_period': 1,
                'max_holding_period': None
            }
        else:
            raise ValueError(f"Unknown timing mode: {mode}")
    
    @staticmethod
    def migrate_legacy_config(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy configuration to new timing format."""
        if 'timing_config' in strategy_config:
            return strategy_config  # Already using new format
        
        # Create timing config from legacy settings
        timing_config = {'mode': 'time_based'}
        
        # Map legacy rebalance_frequency
        if 'rebalance_frequency' in strategy_config:
            timing_config['rebalance_frequency'] = strategy_config['rebalance_frequency']
        
        # Check if strategy has custom daily support (this would need to be determined by strategy type)
        # For now, default to time_based unless explicitly configured
        
        strategy_config['timing_config'] = timing_config
        return strategy_config