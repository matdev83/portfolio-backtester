"""
Trial deduplication for optimization to avoid repeating identical parameter combinations.
"""

import logging
import hashlib
import json
from typing import Dict, Any, Set, Optional
import optuna

logger = logging.getLogger(__name__)


class TrialDeduplicator:
    """
    Manages trial deduplication to avoid running identical parameter combinations.
    
    This is especially important when the parameter space is small and discrete,
    where many trials might end up with the same parameter values.
    """
    
    def __init__(self, enable_deduplication: bool = True):
        self.enable_deduplication = enable_deduplication
        self.seen_parameter_hashes: Set[str] = set()
        self.parameter_to_value: Dict[str, float] = {}
        
    def _hash_parameters(self, parameters: Dict[str, Any]) -> str:
        """
        Create a deterministic hash of parameter values.
        
        Args:
            parameters: Dictionary of parameter names to values
            
        Returns:
            String hash of the parameters
        """
        # Sort parameters by key for deterministic hashing
        sorted_params = dict(sorted(parameters.items()))
        
        # Convert to JSON string and hash
        param_str = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def is_duplicate(self, parameters: Dict[str, Any]) -> bool:
        """
        Check if these parameters have been seen before.
        
        Args:
            parameters: Dictionary of parameter names to values
            
        Returns:
            True if this is a duplicate, False otherwise
        """
        if not self.enable_deduplication:
            return False
            
        param_hash = self._hash_parameters(parameters)
        return param_hash in self.seen_parameter_hashes
    
    def register_parameters(self, parameters: Dict[str, Any], objective_value: Optional[float] = None) -> None:
        """
        Register a parameter combination as seen.
        
        Args:
            parameters: Dictionary of parameter names to values
            objective_value: Optional objective value for this parameter combination
        """
        if not self.enable_deduplication:
            return
            
        param_hash = self._hash_parameters(parameters)
        self.seen_parameter_hashes.add(param_hash)
        
        if objective_value is not None:
            self.parameter_to_value[param_hash] = objective_value
    
    def get_cached_value(self, parameters: Dict[str, Any]) -> Optional[float]:
        """
        Get cached objective value for parameters if available.
        
        Args:
            parameters: Dictionary of parameter names to values
            
        Returns:
            Cached objective value or None if not available
        """
        if not self.enable_deduplication:
            return None
            
        param_hash = self._hash_parameters(parameters)
        return self.parameter_to_value.get(param_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.
        
        Returns:
            Dictionary with deduplication stats
        """
        return {
            'enabled': self.enable_deduplication,
            'unique_parameter_combinations': len(self.seen_parameter_hashes),
            'cached_values': len(self.parameter_to_value)
        }


class DedupOptunaObjectiveAdapter:
    """
    Wrapper around OptunaObjectiveAdapter that adds deduplication capabilities.
    """
    
    def __init__(self, base_objective, enable_deduplication: bool = True):
        self.base_objective = base_objective
        self.deduplicator = TrialDeduplicator(enable_deduplication)
        self.duplicate_count = 0
        self.cache_hits = 0
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Evaluate trial with deduplication.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value
        """
        # Extract parameters from trial
        params = self._trial_to_params(trial)
        
        # Check if we've seen these parameters before
        cached_value = self.deduplicator.get_cached_value(params)
        if cached_value is not None:
            self.cache_hits += 1
            logger.info(f"Trial {trial.number}: Using cached value {cached_value:.6f} for duplicate parameters")
            return cached_value
        
        # Check if this is a duplicate (but no cached value)
        if self.deduplicator.is_duplicate(params):
            self.duplicate_count += 1
            logger.warning(f"Trial {trial.number}: Duplicate parameters detected but no cached value")
        
        # Evaluate using base objective
        objective_value = self.base_objective(trial)
        
        # Register the result
        self.deduplicator.register_parameters(params, objective_value)
        
        return objective_value
    
    def _trial_to_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Extract parameters from trial (delegate to base objective if available).
        """
        if hasattr(self.base_objective, '_trial_to_params'):
            return self.base_objective._trial_to_params(trial)
        
        # Fallback: extract from trial's params (may not be complete)
        return trial.params.copy()
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.
        
        Returns:
            Dictionary with deduplication stats
        """
        base_stats = self.deduplicator.get_stats()
        base_stats.update({
            'duplicate_trials_detected': self.duplicate_count,
            'cache_hits': self.cache_hits
        })
        return base_stats


def create_deduplicating_objective(base_objective, enable_deduplication: bool = True):
    """
    Factory function to create a deduplicating objective wrapper.
    
    Args:
        base_objective: Base objective function to wrap
        enable_deduplication: Whether to enable deduplication
        
    Returns:
        Wrapped objective with deduplication capabilities
    """
    return DedupOptunaObjectiveAdapter(base_objective, enable_deduplication)