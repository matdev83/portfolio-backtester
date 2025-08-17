"""
Property-based tests for deduplication mechanisms.

This module uses Hypothesis to test invariants and properties of the deduplication
mechanisms used to avoid redundant evaluations during optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, cast
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st, assume, example
from hypothesis.extra import numpy as hnp

from portfolio_backtester.optimization.performance.deduplication import (
    BaseTrialDeduplicator,
    GenericTrialDeduplicator
)
from portfolio_backtester.optimization.performance.deduplication_factory import DeduplicationFactory

from tests.strategies.optimization_strategies import (
    parameter_spaces,
    parameter_values,
    populations,
    evaluation_results,
    optimization_configs
)


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(
            st.integers(-100, 100),
            st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
            st.text(min_size=1, max_size=10),
            st.booleans()
        ),
        min_size=1,
        max_size=5
    )
)
@settings(deadline=None)
def test_deduplicator_hash_parameters(params):
    """Test that parameter hashing is deterministic."""
    # Create a deduplicator
    deduplicator = GenericTrialDeduplicator()
    
    # Hash the parameters
    hash1 = deduplicator._hash_parameters(params)
    hash2 = deduplicator._hash_parameters(params)
    
    # Check that the hash is deterministic
    assert hash1 == hash2
    assert isinstance(hash1, str)
    
    # Check that different parameter orders produce the same hash
    params_copy = {}
    items = list(params.items())
    # Reverse the order of items
    for k, v in reversed(items):
        params_copy[k] = v
    
    hash3 = deduplicator._hash_parameters(params_copy)
    assert hash1 == hash3


@given(
    st.lists(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(
                st.integers(-100, 100),
                st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                st.text(min_size=1, max_size=10),
                st.booleans()
            ),
            min_size=1,
            max_size=5
        ),
        min_size=2,
        max_size=10
    )
)
@settings(deadline=None)
def test_deduplicator_duplicate_detection(param_list):
    """Test that the deduplicator correctly identifies duplicates."""
    # Create a deduplicator
    deduplicator = GenericTrialDeduplicator(enable_deduplication=True)
    
    # Add the first set of parameters
    deduplicator.add_trial(param_list[0])
    
    # Check that the first set is now considered a duplicate
    assert deduplicator.is_duplicate(param_list[0])
    
    # Check that other parameter sets are not duplicates (unless they happen to be identical)
    for i in range(1, len(param_list)):
        is_duplicate = deduplicator.is_duplicate(param_list[i])
        
        # If it's a duplicate, verify that it's actually identical to a previous set
        if is_duplicate:
            hash_i = deduplicator._hash_parameters(param_list[i])
            hash_0 = deduplicator._hash_parameters(param_list[0])
            assert hash_i == hash_0
    
    # Add all parameter sets
    for params in param_list:
        deduplicator.add_trial(params)
    
    # Check that all parameter sets are now duplicates
    for params in param_list:
        assert deduplicator.is_duplicate(params)


@given(
    st.lists(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(
                st.integers(-100, 100),
                st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                st.text(min_size=1, max_size=10),
                st.booleans()
            ),
            min_size=1,
            max_size=5
        ),
        min_size=2,
        max_size=10,
        unique_by=lambda x: frozenset(x.items())
    ),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(deadline=None)
def test_deduplicator_cached_values(param_list, result_value):
    """Test that the deduplicator correctly caches and retrieves values."""
    # Create a deduplicator
    deduplicator = GenericTrialDeduplicator(enable_deduplication=True)
    
    # Add the first set of parameters with a result
    deduplicator.add_trial(param_list[0], result_value)
    
    # Check that the cached value is correct
    assert deduplicator.get_cached_value(param_list[0]) == result_value
    
    # Check that other parameter sets don't have cached values
    for i in range(1, len(param_list)):
        cached_value = deduplicator.get_cached_value(param_list[i])
        
        # If there's a cached value, verify that it's because the parameters are identical
        if cached_value is not None:
            hash_i = deduplicator._hash_parameters(param_list[i])
            hash_0 = deduplicator._hash_parameters(param_list[0])
            assert hash_i == hash_0
            assert cached_value == result_value


@given(
    st.booleans(),
    st.sampled_from(["optuna", "genetic", "bayesian", "random"])
)
@settings(deadline=None)
def test_deduplication_factory(enable_deduplication, optimizer_type):
    """Test that the deduplication factory creates the correct deduplicator."""
    # Create a deduplicator using the factory
    config = {"enable_deduplication": enable_deduplication}
    deduplicator = DeduplicationFactory.create_deduplicator(optimizer_type, config)
    
    # Check that the deduplicator has the correct attributes
    assert deduplicator.enable_deduplication == enable_deduplication
    
    # Check that the deduplicator is the correct type
    assert isinstance(deduplicator, GenericTrialDeduplicator)


@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["param_a", "param_b", "param_c"]),
            values=st.integers(0, 10),
            min_size=1,
            max_size=3
        ),
        min_size=2,
        max_size=10
    )
)
@settings(deadline=None)
def test_deduplicator_stats(param_list):
    """Test that the deduplicator correctly tracks statistics."""
    # Create a deduplicator
    deduplicator = GenericTrialDeduplicator(enable_deduplication=True)
    
    # Add all parameter sets
    unique_hashes = set()
    for params in param_list:
        deduplicator.add_trial(params)
        unique_hashes.add(deduplicator._hash_parameters(params))
    
    # Get the stats
    stats = deduplicator.get_stats()
    
    # Check that the stats are correct
    assert stats["enabled"] == True
    assert stats["unique_parameter_combinations"] == len(unique_hashes)
    assert stats["cached_values"] == 0  # No values were cached
    
    # Add some parameter sets with results
    for i, params in enumerate(param_list):
        deduplicator.add_trial(params, float(i))
    
    # Get the stats again
    stats = deduplicator.get_stats()
    
    # Check that the stats are updated
    assert stats["enabled"] == True
    assert stats["unique_parameter_combinations"] == len(unique_hashes)
    assert stats["cached_values"] == len(unique_hashes)  # All unique hashes should have cached values


@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["param_a", "param_b", "param_c"]),
            values=st.integers(0, 10),
            min_size=1,
            max_size=3
        ),
        min_size=2,
        max_size=10
    )
)
@settings(deadline=None)
def test_deduplicator_disabled(param_list):
    """Test that the deduplicator correctly handles being disabled."""
    # Create a deduplicator with deduplication disabled
    deduplicator = GenericTrialDeduplicator(enable_deduplication=False)
    
    # Add all parameter sets
    for params in param_list:
        deduplicator.add_trial(params)
    
    # Check that no parameter sets are considered duplicates
    for params in param_list:
        assert not deduplicator.is_duplicate(params)
    
    # Check that no parameter sets have cached values
    for params in param_list:
        assert deduplicator.get_cached_value(params) is None
    
    # Get the stats
    stats = deduplicator.get_stats()
    
    # Check that the stats reflect that deduplication is disabled
    assert stats["enabled"] == False
