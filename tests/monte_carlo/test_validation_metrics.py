"""
Tests for Validation Metrics

This module contains comprehensive tests for the synthetic data validation
functionality, including statistical tests and quality assessment.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import logging

from src.portfolio_backtester.monte_carlo.validation_metrics import (
    SyntheticDataValidator,
    ValidationResults
)


class TestValidationResults:
    """Test suite for ValidationResults dataclass."""
    
    def test_validation_results_creation(self):
        """Test validation results creation."""
        result = ValidationResults(
            test_name="test_example",
            passed=True,
            p_value=0.05,
            statistic=1.96,
            critical_value=1.64,
            details={'key': 'value'}
        )
        
        assert result.test_name == "test_example"
        assert result.passed is True
        assert result.p_value == 0.05
        assert result.statistic == 1.96
        assert result.critical_value == 1.64
        assert result.details == {'key': 'value'}
    
    def test_validation_results_minimal(self):
        """Test validation results with minimal fields."""
        result = ValidationResults(
            test_name="minimal_test",
            passed=False
        )
        
        assert result.test_name == "minimal_test"
        assert result.passed is False
        assert result.p_value is None
        assert result.statistic is None
        assert result.critical_value is None
        assert result.details is None


class TestSyntheticDataValidator:
    """Test suite for SyntheticDataValidator class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'validation_config': {
                'basic_stats_tolerance': 0.2,
                'ks_test_pvalue_threshold': 0.05,
                'autocorr_max_deviation': 0.1,
                'volatility_clustering_threshold': 0.05,
                'tail_index_tolerance': 0.5,
                'extreme_value_tolerance': 0.5,
                'overall_quality_threshold': 0.7
            }
        }
    
    @pytest.fixture
    def sample_original_data(self):
        """Generate sample original data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Generate returns with some realistic properties
        returns = []
        volatility = 0.02
        
        for i in range(len(dates)):
            if i > 0:
                # Add some volatility clustering
                volatility = 0.0001 + 0.05 * (returns[i-1]**2) + 0.9 * volatility
            
            daily_return = np.random.normal(0, np.sqrt(volatility))
            returns.append(daily_return)
        
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def sample_synthetic_data(self, sample_original_data):
        """Generate sample synthetic data similar to original."""
        np.random.seed(123)  # Different seed for variation
        
        # Create synthetic data with similar properties
        original_stats = {
            'mean': sample_original_data.mean(),
            'std': sample_original_data.std(),
            'skew': sample_original_data.skew(),
            'kurt': sample_original_data.kurtosis()
        }
        
        # Generate synthetic data with similar properties
        synthetic_returns = np.random.normal(
            original_stats['mean'], 
            original_stats['std'], 
            len(sample_original_data)
        )
        
        return pd.Series(synthetic_returns, index=sample_original_data.index)
    
    def test_initialization(self, sample_config):
        """Test validator initialization."""
        validator = SyntheticDataValidator(sample_config['validation_config'])
        
        assert validator.validation_config == sample_config['validation_config']
        assert validator.results_history == []
    
    def test_validate_basic_statistics_pass(self, sample_config):
        """Test basic statistics validation that passes."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create similar data
        np.random.seed(42)
        original = pd.Series(np.random.normal(0, 1, 1000))
        
        np.random.seed(43)  # Slightly different but similar
        synthetic = pd.Series(np.random.normal(0, 1, 1000))
        
        result = validator._validate_basic_statistics(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "basic_statistics"
        assert result.details is not None
        assert 'original_stats' in result.details
        assert 'synthetic_stats' in result.details
        assert 'relative_differences' in result.details
    
    def test_validate_basic_statistics_fail(self, sample_config):
        """Test basic statistics validation that fails."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create very different data
        original = pd.Series(np.random.normal(0, 1, 1000))
        synthetic = pd.Series(np.random.normal(10, 0.1, 1000))  # Very different
        
        result = validator._validate_basic_statistics(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "basic_statistics"
        assert result.passed is False
    
    def test_validate_distribution_similarity(self, sample_config):
        """Test distribution similarity validation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create similar distributions
        np.random.seed(42)
        original = pd.Series(np.random.normal(0, 1, 1000))
        synthetic = pd.Series(np.random.normal(0, 1, 1000))
        
        result = validator._validate_distribution_similarity(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "distribution_similarity"
        assert result.p_value is not None
        assert result.statistic is not None
        assert result.details is not None
    
    def test_validate_distribution_similarity_insufficient_data(self, sample_config):
        """Test distribution similarity with insufficient data."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create very short series
        original = pd.Series([1, 2, 3])
        synthetic = pd.Series([1.1, 2.1, 3.1])
        
        result = validator._validate_distribution_similarity(original, synthetic, "TEST")
        
        assert result.passed is False
        assert result.details is not None
        assert 'error' in result.details # Ensure 'error' key is present
    
    def test_validate_autocorrelation_structure(self, sample_config):
        """Test autocorrelation structure validation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create data with some autocorrelation
        np.random.seed(42)
        original_base = np.random.normal(0, 1, 1000)
        original = pd.Series([original_base[0]] + [0.1 * original_base[i-1] + 0.9 * original_base[i] for i in range(1, 1000)])
        
        np.random.seed(43)
        synthetic_base = np.random.normal(0, 1, 1000)
        synthetic = pd.Series([synthetic_base[0]] + [0.1 * synthetic_base[i-1] + 0.9 * synthetic_base[i] for i in range(1, 1000)])
        
        result = validator._validate_autocorrelation_structure(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "autocorrelation_structure"
        assert result.details is not None
        assert 'max_difference' in result.details
        assert 'avg_difference' in result.details
    
    def test_validate_volatility_clustering(self, sample_config):
        """Test volatility clustering validation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create data with volatility clustering
        np.random.seed(42)
        returns = []
        volatility = 0.02
        
        for i in range(500):
            if i > 0:
                volatility = 0.0001 + 0.05 * (returns[i-1]**2) + 0.9 * volatility
            
            daily_return = np.random.normal(0, np.sqrt(volatility))
            returns.append(daily_return)
        
        original = pd.Series(returns)
        
        # Create synthetic with similar clustering
        np.random.seed(43)
        synthetic_returns = []
        volatility = 0.02
        
        for i in range(500):
            if i > 0:
                volatility = 0.0001 + 0.05 * (synthetic_returns[i-1]**2) + 0.9 * volatility
            
            daily_return = np.random.normal(0, np.sqrt(volatility))
            synthetic_returns.append(daily_return)
        
        synthetic = pd.Series(synthetic_returns)
        
        result = validator._validate_volatility_clustering(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "volatility_clustering"
        assert result.details is not None
        assert 'original_lag1_autocorr' in result.details
        assert 'synthetic_lag1_autocorr' in result.details
    
    def test_validate_fat_tails(self, sample_config):
        """Test fat tail validation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create data with fat tails (Student-t distribution)
        np.random.seed(42)
        original = pd.Series(np.random.standard_t(df=3, size=1000))
        
        np.random.seed(43)
        synthetic = pd.Series(np.random.standard_t(df=3, size=1000))
        
        result = validator._validate_fat_tails(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "fat_tails"
        assert result.details is not None
        assert 'original_tail_index' in result.details
        assert 'synthetic_tail_index' in result.details
    
    def test_validate_extreme_values(self, sample_config):
        """Test extreme value validation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create similar data
        np.random.seed(42)
        original = pd.Series(np.random.normal(0, 1, 1000))
        
        np.random.seed(43)
        synthetic = pd.Series(np.random.normal(0, 1, 1000))
        
        result = validator._validate_extreme_values(original, synthetic, "TEST")
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "extreme_values"
        assert result.details is not None
        assert 'original_percentiles' in result.details
        assert 'synthetic_percentiles' in result.details
    
    def test_calculate_overall_quality(self, sample_config):
        """Test overall quality calculation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create mock results
        mock_results = {
            'test1': ValidationResults("test1", True),
            'test2': ValidationResults("test2", False),
            'test3': ValidationResults("test3", True),
            'test4': ValidationResults("test4", True)
        }
        
        result = validator._calculate_overall_quality(mock_results)
        
        assert isinstance(result, ValidationResults)
        assert result.test_name == "overall_quality"
        assert result.details is not None
        assert 'quality_score' in result.details
        assert 'passed_tests' in result.details
        assert 'total_tests' in result.details
        
        # Should be 3/4 = 0.75
        assert result.details['quality_score'] == 0.75
        assert result.details['passed_tests'] == 3
        assert result.details['total_tests'] == 4
    
    def test_estimate_tail_index(self, sample_config):
        """Test tail index estimation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Test with Student-t data (known fat tails)
        np.random.seed(42)
        fat_tail_data = pd.Series(np.random.standard_t(df=3, size=1000))
        
        tail_index = validator._estimate_tail_index(fat_tail_data)
        
        assert isinstance(tail_index, float)
        assert 1.5 <= tail_index <= 10.0  # Should be bounded
    
    def test_estimate_tail_index_insufficient_data(self, sample_config):
        """Test tail index estimation with insufficient data."""
        validator = SyntheticDataValidator(sample_config)
        
        # Very short series
        short_data = pd.Series([1, 2, 3])
        
        tail_index = validator._estimate_tail_index(short_data)
        
        assert tail_index == 3.0  # Default fallback
    
    def test_validate_synthetic_data_comprehensive(self, sample_config, sample_original_data, sample_synthetic_data):
        """Test comprehensive validation of synthetic data."""
        validator = SyntheticDataValidator(sample_config)
        
        results = validator.validate_synthetic_data(
            sample_original_data, 
            sample_synthetic_data, 
            "TEST_ASSET"
        )
        
        # Check all expected tests are present
        expected_tests = [
            'basic_stats', 'distribution', 'autocorrelation', 
            'volatility_clustering', 'fat_tails', 'extreme_values', 
            'overall_quality'
        ]
        
        for test_name in expected_tests:
            assert test_name in results
            assert isinstance(results[test_name], ValidationResults)
        
        # Check that results are stored in history
        assert len(validator.results_history) > 0
    
    def test_generate_validation_report(self, sample_config):
        """Test validation report generation."""
        validator = SyntheticDataValidator(sample_config)
        
        # Create mock results
        mock_results = {
            'basic_stats': ValidationResults("basic_stats", True, details={'key': 'value'}),
            'distribution': ValidationResults("distribution", False, p_value=0.01),
            'overall_quality': ValidationResults("overall_quality", True, details={'quality_score': 0.8})
        }
        
        report = validator.generate_validation_report(mock_results, "TEST_ASSET")
        
        assert isinstance(report, str)
        assert "TEST_ASSET" in report
        assert "Overall Quality" in report
        assert "PASS" in report
        assert "FAIL" in report
    
    def test_get_validation_summary_empty(self, sample_config):
        """Test validation summary with no history."""
        validator = SyntheticDataValidator(sample_config)
        
        summary = validator.get_validation_summary()
        
        assert summary == {"total_validations": 0}
    
    def test_get_validation_summary_with_data(self, sample_config):
        """Test validation summary with validation history."""
        validator = SyntheticDataValidator(sample_config)
        
        # Add some mock results to history
        validator.results_history = [
            ValidationResults("test1", True),
            ValidationResults("test1", False),
            ValidationResults("test2", True),
            ValidationResults("test2", True)
        ]
        
        summary = validator.get_validation_summary()
        
        assert summary['total_validations'] == 4
        assert summary['passed_validations'] == 3
        assert summary['overall_pass_rate'] == 0.75
        assert 'test_type_statistics' in summary
        
        # Check test type statistics
        test_stats = summary['test_type_statistics']
        assert 'test1' in test_stats
        assert 'test2' in test_stats
        assert test_stats['test1']['total'] == 2
        assert test_stats['test1']['passed'] == 1
        assert test_stats['test1']['pass_rate'] == 0.5
        assert test_stats['test2']['total'] == 2
        assert test_stats['test2']['passed'] == 2
        assert test_stats['test2']['pass_rate'] == 1.0


# Integration tests
class TestValidationIntegration:
    """Integration tests for validation with realistic data."""
    
    @pytest.fixture
    def realistic_config(self):
        """Realistic configuration for integration testing."""
        return {
            'validation_config': {
                'basic_stats_tolerance': 0.3,
                'ks_test_pvalue_threshold': 0.01,
                'autocorr_max_deviation': 0.15,
                'volatility_clustering_threshold': 0.03,
                'tail_index_tolerance': 0.7,
                'extreme_value_tolerance': 0.7,
                'overall_quality_threshold': 0.6
            }
        }
    
    @pytest.fixture
    def realistic_original_data(self):
        """Generate realistic financial returns data."""
        np.random.seed(42)
        
        # Generate GARCH-like returns
        returns = []
        volatility = 0.02
        
        for i in range(2000):
            if i > 0:
                # Volatility clustering
                volatility = 0.0001 + 0.05 * (returns[i-1]**2) + 0.9 * volatility
            
            # Use Student-t for fat tails
            daily_return = np.random.standard_t(df=5) * np.sqrt(volatility)
            returns.append(daily_return)
        
        dates = pd.date_range('2020-01-01', periods=len(returns), freq='D')
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def realistic_synthetic_data(self, realistic_original_data):
        """Generate realistic synthetic data using similar process."""
        np.random.seed(123)  # Different seed
        
        # Estimate parameters from original data
        orig_std = realistic_original_data.std()
        orig_mean = realistic_original_data.mean()
        
        # Generate synthetic with similar GARCH process
        synthetic_returns = []
        volatility = 0.02
        
        for i in range(len(realistic_original_data)):
            if i > 0:
                volatility = 0.0001 + 0.05 * (synthetic_returns[i-1]**2) + 0.9 * volatility
            
            daily_return = np.random.standard_t(df=5) * np.sqrt(volatility)
            synthetic_returns.append(daily_return)
        
        # Adjust to match original mean/std approximately
        synthetic_series = pd.Series(synthetic_returns, index=realistic_original_data.index)
        synthetic_series = (synthetic_series - synthetic_series.mean()) / synthetic_series.std()
        synthetic_series = synthetic_series * orig_std + orig_mean
        
        return synthetic_series
    
    def test_realistic_validation_comprehensive(self, realistic_config, realistic_original_data, realistic_synthetic_data):
        """Test comprehensive validation with realistic data."""
        validator = SyntheticDataValidator(realistic_config)
        
        results = validator.validate_synthetic_data(
            realistic_original_data,
            realistic_synthetic_data,
            "REALISTIC_ASSET"
        )
        
        # Check that all tests completed
        expected_tests = [
            'basic_stats', 'distribution', 'autocorrelation', 
            'volatility_clustering', 'fat_tails', 'extreme_values', 
            'overall_quality'
        ]
        
        for test_name in expected_tests:
            assert test_name in results
            assert isinstance(results[test_name], ValidationResults)
        
        # Generate and check report
        report = validator.generate_validation_report(results, "REALISTIC_ASSET")
        assert len(report) > 0
        assert "REALISTIC_ASSET" in report
    
    def test_validation_with_poor_synthetic_data(self, realistic_config, realistic_original_data):
        """Test validation with poor quality synthetic data."""
        validator = SyntheticDataValidator(realistic_config)
        
        # Create poor quality synthetic data (just random normal)
        np.random.seed(999)
        poor_synthetic = pd.Series(
            np.random.normal(0, 1, len(realistic_original_data)),
            index=realistic_original_data.index
        )
        
        results = validator.validate_synthetic_data(
            realistic_original_data,
            poor_synthetic,
            "POOR_SYNTHETIC"
        )
        
        # Overall quality should be poor
        overall_quality = results['overall_quality']
        if overall_quality.details and 'quality_score' in overall_quality.details:
            assert overall_quality.details['quality_score'] < 0.7  # Should fail overall
        else:
            pytest.fail(f"Overall quality details or quality_score missing: {overall_quality.details}")
    
    def test_validation_summary_multiple_assets(self, realistic_config, realistic_original_data, realistic_synthetic_data):
        """Test validation summary with multiple assets."""
        validator = SyntheticDataValidator(realistic_config)
        
        # Validate multiple "assets"
        for i in range(3):
            # Add some noise to make each validation slightly different
            noisy_synthetic = realistic_synthetic_data + np.random.normal(0, 0.001, len(realistic_synthetic_data))
            
            validator.validate_synthetic_data(
                realistic_original_data,
                noisy_synthetic,
                f"ASSET_{i}"
            )
        
        # Get summary
        summary = validator.get_validation_summary()
        
        assert summary['total_validations'] > 0
        assert 'test_type_statistics' in summary
        assert summary['overall_pass_rate'] >= 0.0
        assert summary['overall_pass_rate'] <= 1.0
    
    def test_edge_case_identical_data(self, realistic_config, realistic_original_data):
        """Test validation with identical original and synthetic data."""
        validator = SyntheticDataValidator(realistic_config)
        
        # Use identical data
        results = validator.validate_synthetic_data(
            realistic_original_data,
            realistic_original_data,  # Same data
            "IDENTICAL"
        )
        
        # Most tests should pass with identical data
        overall_quality = results['overall_quality']
        if overall_quality.details and 'quality_score' in overall_quality.details:
            assert overall_quality.details['quality_score'] > 0.8  # Should score highly
        else:
            pytest.fail(f"Overall quality details or quality_score missing: {overall_quality.details}")
    
    def test_edge_case_very_short_data(self, realistic_config):
        """Test validation with very short data series."""
        validator = SyntheticDataValidator(realistic_config)
        
        # Very short series
        short_original = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008])
        short_synthetic = pd.Series([0.012, -0.018, 0.014, -0.006, 0.007])
        
        results = validator.validate_synthetic_data(
            short_original,
            short_synthetic,
            "SHORT_DATA"
        )
        
        # Should handle gracefully (some tests may fail due to insufficient data)
        assert 'overall_quality' in results
        assert isinstance(results['overall_quality'], ValidationResults) 