import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.signal_generators import (
    MomentumSignalGenerator,
    SharpeSignalGenerator,
    SortinoSignalGenerator,
    CalmarSignalGenerator,
    VAMSSignalGenerator,
    DPVAMSSignalGenerator,
)
from src.portfolio_backtester.feature import (
    Momentum,
    SharpeRatio,
    SortinoRatio,
    CalmarRatio,
    VAMS,
    DPVAMS,
)

# Test data
mock_features_dict = {
    "momentum_6m": pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}),
    "momentum_12m": pd.DataFrame({"A": [0.15, 0.25], "B": [0.35, 0.45]}),
    "sharpe_6m": pd.DataFrame({"A": [1.0, 1.2], "B": [1.3, 1.4]}),
    "sharpe_12m": pd.DataFrame({"A": [1.1, 1.3], "B": [1.4, 1.5]}),
    "sortino_6m": pd.DataFrame({"A": [1.5, 1.7], "B": [1.8, 1.9]}),
    "sortino_12m": pd.DataFrame({"A": [1.6, 1.8], "B": [1.9, 2.0]}),
    "calmar_6m": pd.DataFrame({"A": [0.5, 0.6], "B": [0.7, 0.8]}),
    "calmar_12m": pd.DataFrame({"A": [0.55, 0.65], "B": [0.75, 0.85]}),
    "vams_6m": pd.DataFrame({"A": [2.0, 2.1], "B": [2.2, 2.3]}),
    "vams_12m": pd.DataFrame({"A": [2.1, 2.2], "B": [2.3, 2.4]}),
    "dp_vams_6m_0.50a": pd.DataFrame({"A": [3.0, 3.1], "B": [3.2, 3.3]}),
    "dp_vams_12m_0.50a": pd.DataFrame({"A": [3.1, 3.2], "B": [3.3, 3.4]}),
    "dp_vams_6m_0.70a": pd.DataFrame({"A": [3.2, 3.3], "B": [3.4, 3.5]}),
}

class_params = [
    (MomentumSignalGenerator, "lookback_months", Momentum, "momentum"),
    (SharpeSignalGenerator, "rolling_window", SharpeRatio, "sharpe"),
    (SortinoSignalGenerator, "rolling_window", SortinoRatio, "sortino"),
    (CalmarSignalGenerator, "rolling_window", CalmarRatio, "calmar"),
    (VAMSSignalGenerator, "lookback_months", VAMS, "vams"),
]

@pytest.mark.parametrize("generator_class, param_name, feature_class, feature_prefix", class_params)
def test_signal_generator_required_features_no_optimize(generator_class, param_name, feature_class, feature_prefix):
    config = {"strategy_params": {param_name: 6}}
    generator = generator_class(config)
    features = generator.required_features()
    assert len(features) == 1
    assert feature_class(6) in features

@pytest.mark.parametrize("generator_class, param_name, feature_class, feature_prefix", class_params)
def test_signal_generator_required_features_with_optimize(generator_class, param_name, feature_class, feature_prefix):
    config = {
        "strategy_params": {param_name: 6},
        "optimize": [
            {"parameter": param_name, "min_value": 3, "max_value": 9, "step": 3},
            {"parameter": "other_param", "min_value": 1, "max_value": 5, "step": 1}, # Should be ignored
        ],
    }
    if generator_class == SortinoSignalGenerator:
        config["strategy_params"]["target_return"] = 0.0

    generator = generator_class(config)
    features = generator.required_features()

    expected_features = {feature_class(6)} # from strategy_params
    if generator_class == SortinoSignalGenerator:
        expected_features = {feature_class(6, target_return=0.0)}


    for val in np.arange(3, 9 + 3, 3):
        if generator_class == SortinoSignalGenerator:
            expected_features.add(feature_class(int(val), target_return=0.0))
        else:
            expected_features.add(feature_class(int(val)))

    assert features == expected_features

@pytest.mark.parametrize("generator_class, param_name, feature_class, feature_prefix", class_params)
def test_signal_generator_scores(generator_class, param_name, feature_class, feature_prefix):
    config = {"strategy_params": {param_name: 6}}
    if generator_class == SortinoSignalGenerator:
        config["strategy_params"]["target_return"] = 0.0
    generator = generator_class(config)
    scores = generator.scores(mock_features_dict)
    expected_key = f"{feature_prefix}_6m"
    pd.testing.assert_frame_equal(scores, mock_features_dict[expected_key])

@pytest.mark.parametrize("generator_class, param_name, feature_class, feature_prefix", class_params)
def test_signal_generator_scores_different_param(generator_class, param_name, feature_class, feature_prefix):
    config = {"strategy_params": {param_name: 12}}
    if generator_class == SortinoSignalGenerator:
        config["strategy_params"]["target_return"] = 0.0
    generator = generator_class(config)
    scores = generator.scores(mock_features_dict)
    expected_key = f"{feature_prefix}_12m"
    pd.testing.assert_frame_equal(scores, mock_features_dict[expected_key])


# Tests for DPVAMSSignalGenerator
def test_dpvams_signal_generator_required_features_no_optimize():
    config = {"strategy_params": {"lookback_months": 6, "alpha": 0.5}}
    generator = DPVAMSSignalGenerator(config)
    features = generator.required_features()
    assert DPVAMS(lookback_months=6, alpha="0.50") in features
    assert len(features) == 1

def test_dpvams_signal_generator_required_features_with_optimize_lookback():
    config = {
        "strategy_params": {"lookback_months": 6, "alpha": 0.5},
        "optimize": [
            {"parameter": "lookback_months", "min_value": 3, "max_value": 9, "step": 3},
            {"parameter": "other_param", "min_value": 1, "max_value": 5, "step": 1},
        ],
    }
    generator = DPVAMSSignalGenerator(config)
    features = generator.required_features()
    expected = {
        DPVAMS(lookback_months=6, alpha="0.50"), # from strategy_params
        DPVAMS(lookback_months=3, alpha="0.50"),
        DPVAMS(lookback_months=6, alpha="0.50"),
        DPVAMS(lookback_months=9, alpha="0.50"),
    }
    assert features == expected

def test_dpvams_signal_generator_required_features_with_optimize_alpha():
    config = {
        "strategy_params": {"lookback_months": 6, "alpha": 0.5},
        "optimize": [
            {"parameter": "alpha", "min_value": 0.3, "max_value": 0.7, "step": 0.2},
            {"parameter": "other_param", "min_value": 1, "max_value": 5, "step": 1},
        ],
    }
    generator = DPVAMSSignalGenerator(config)
    features = generator.required_features()
    # Using np.isclose for comparing float-based alpha strings
    generated_alphas = sorted([feat.alpha for feat in features if feat.lookback_months == 6])
    expected_alphas_values = [0.3, 0.5, 0.7] # 0.5 from strategy_params, 0.3, 0.5, 0.7 from optimize

    # Check if all expected alphas are present
    present_alphas_from_optimize = sorted([f"{val:.2f}" for val in np.arange(0.3, 0.7 + 0.2, 0.2)])
    expected_alpha_set = set([DPVAMS(lookback_months=6, alpha=a) for a in present_alphas_from_optimize])
    expected_alpha_set.add(DPVAMS(lookback_months=6, alpha="0.50")) # from strategy_params

    assert features == expected_alpha_set


def test_dpvams_signal_generator_required_features_with_optimize_both():
    config = {
        "strategy_params": {"lookback_months": 6, "alpha": 0.5},
        "optimize": [
            {"parameter": "lookback_months", "min_value": 3, "max_value": 4, "step": 1},
            {"parameter": "alpha", "min_value": 0.6, "max_value": 0.7, "step": 0.1},
        ],
    }
    generator = DPVAMSSignalGenerator(config)
    features = generator.required_features()

    expected = {
        DPVAMS(lookback_months=6, alpha="0.50"), # strategy_params
        DPVAMS(lookback_months=3, alpha="0.50"), # optimize lookback, strategy alpha
        DPVAMS(lookback_months=4, alpha="0.50"), # optimize lookback, strategy alpha
        DPVAMS(lookback_months=6, alpha="0.60"), # strategy lookback, optimize alpha
        DPVAMS(lookback_months=6, alpha="0.70"), # strategy lookback, optimize alpha
    }
    assert features == expected


def test_dpvams_signal_generator_scores():
    config = {"strategy_params": {"lookback_months": 6, "alpha": 0.5}}
    generator = DPVAMSSignalGenerator(config)
    scores = generator.scores(mock_features_dict)
    pd.testing.assert_frame_equal(scores, mock_features_dict["dp_vams_6m_0.50a"])

def test_dpvams_signal_generator_scores_different_params():
    config = {"strategy_params": {"lookback_months": 12, "alpha": 0.5}}
    generator = DPVAMSSignalGenerator(config)
    scores = generator.scores(mock_features_dict)
    pd.testing.assert_frame_equal(scores, mock_features_dict["dp_vams_12m_0.50a"])

def test_dpvams_signal_generator_scores_different_alpha():
    config = {"strategy_params": {"lookback_months": 6, "alpha": 0.7}} # Alpha as float
    generator = DPVAMSSignalGenerator(config)
    scores = generator.scores(mock_features_dict)
    pd.testing.assert_frame_equal(scores, mock_features_dict["dp_vams_6m_0.70a"])

def test_signal_generator_required_features_empty_config():
    # Test that if strategy_params is missing, it defaults gracefully or handles as expected
    # For MomentumSignalGenerator, if 'lookback_months' is not in params, it should not add a feature
    # If 'optimize' is also not there, it should return an empty set.
    generator = MomentumSignalGenerator({}) # Empty config
    features = generator.required_features()
    assert len(features) == 0

    generator_sharpe = SharpeSignalGenerator({})
    features_sharpe = generator_sharpe.required_features()
    assert len(features_sharpe) == 0

    # DPVAMS requires both lookback_months and alpha
    generator_dpvams = DPVAMSSignalGenerator({})
    features_dpvams = generator_dpvams.required_features()
    assert len(features_dpvams) == 0

    generator_dpvams_partial1 = DPVAMSSignalGenerator({"strategy_params": {"lookback_months": 6}})
    features_dpvams_partial1 = generator_dpvams_partial1.required_features()
    assert len(features_dpvams_partial1) == 0

    generator_dpvams_partial2 = DPVAMSSignalGenerator({"strategy_params": {"alpha": 0.5}})
    features_dpvams_partial2 = generator_dpvams_partial2.required_features()
    assert len(features_dpvams_partial2) == 0


@pytest.mark.parametrize("generator_class, param_name, feature_class, feature_prefix", class_params)
def test_signal_generator_required_features_optimize_no_step(generator_class, param_name, feature_class, feature_prefix):
    # Test optimize case where step is not provided (should default to 1)
    config = {
        "strategy_params": {param_name: 6},
        "optimize": [
            {"parameter": param_name, "min_value": 3, "max_value": 5}, # No step
        ],
    }
    if generator_class == SortinoSignalGenerator:
        config["strategy_params"]["target_return"] = 0.0

    generator = generator_class(config)
    features = generator.required_features()

    expected_features = {feature_class(6)}
    if generator_class == SortinoSignalGenerator:
        expected_features = {feature_class(6, target_return=0.0)}


    for val in np.arange(3, 5 + 1, 1): # Default step is 1
        if generator_class == SortinoSignalGenerator:
            expected_features.add(feature_class(int(val), target_return=0.0))
        else:
            expected_features.add(feature_class(int(val)))

    assert features == expected_features

def test_dpvams_required_features_optimize_no_step():
    config = {
        "strategy_params": {"lookback_months": 6, "alpha": 0.5},
        "optimize": [
            {"parameter": "lookback_months", "min_value": 3, "max_value": 4}, # No step for lookback
            {"parameter": "alpha", "min_value": 0.6, "max_value": 0.7, "step": 0.1},
        ],
    }
    generator = DPVAMSSignalGenerator(config)
    features = generator.required_features()

    expected = {
        DPVAMS(lookback_months=6, alpha="0.50"),
        DPVAMS(lookback_months=3, alpha="0.50"),
        DPVAMS(lookback_months=4, alpha="0.50"),
        DPVAMS(lookback_months=6, alpha="0.60"),
        DPVAMS(lookback_months=6, alpha="0.70"),
    }
    assert features == expected

    config_alpha_no_step = {
        "strategy_params": {"lookback_months": 6, "alpha": 0.5},
        "optimize": [
            {"parameter": "lookback_months", "min_value": 3, "max_value": 4, "step": 1},
            {"parameter": "alpha", "min_value": 0.6, "max_value": 0.7}, # No step for alpha, should still generate 0.60 and 0.70 if values are distinct enough
        ],
    }
    generator_alpha_no_step = DPVAMSSignalGenerator(config_alpha_no_step)
    features_alpha_no_step = generator_alpha_no_step.required_features()

    # If step for alpha is 1 (default), only min_value=0.6 would be added for alpha.
    # The intention of np.arange(min, max + step, step) is tricky if step is not small for floats.
    # For float params without step, it's safer to assume it might only take min_value or behave unexpectedly if max_value isn't reachable by adding integer steps.
    # Given current loop `np.arange(min_v, max_v + step, step)` and step defaulting to 1:
    # For alpha: np.arange(0.6, 0.7 + 1, 1) -> [0.6]. So only alpha=0.60 from optimize.
    expected_alpha_no_step = {
            DPVAMS(lookback_months=6, alpha="0.50"),
            DPVAMS(lookback_months=3, alpha="0.50"),
            DPVAMS(lookback_months=4, alpha="0.50"),
            DPVAMS(lookback_months=6, alpha="0.60"), # from val=0.6
            DPVAMS(lookback_months=6, alpha="1.60"), # from val=1.6 due to np.arange(0.6, 0.7 + 1, 1) giving [0.6, 1.6]
    }
    assert features_alpha_no_step == expected_alpha_no_step
