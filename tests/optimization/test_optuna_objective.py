import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.optimization.optuna_objective import build_objective


# Common setup for tests
@pytest.fixture
def common_mocks():
    g_cfg = {"benchmark": "SPY"}
    train_data_monthly = MagicMock()
    train_data_daily = MagicMock()
    train_rets_daily = MagicMock()
    bench_series_daily = MagicMock()
    features_slice = MagicMock()
    trial = MagicMock()

    # Extended OPTIMIZER_PARAMETER_DEFAULTS for new tests
    optimizer_defaults = {
        "max_lookback": {"type": "int", "low": 20, "high": 252, "step": 10},
        "leverage": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.1, "log": False}, # step can be None for float
        "numeric_enum": {"type": "categorical", "values": [10, 20, 30]},
        "string_enum": {"type": "categorical", "values": ["valA", "valB"]},
        "position_sizer": { # Example for SPECIAL_SCEN_CFG_KEYS
            "type": "categorical",
            "values": ["equal_weight", "rolling_sharpe"]
        },
        "param_to_override": {"type": "int", "low": 1, "high": 5, "step": 1},
        "default_choices_param": {"type": "categorical", "values": ["default1", "default2"]},
    }

    return {
        "g_cfg": g_cfg,
        "train_data_monthly": train_data_monthly,
        "train_data_daily": train_data_daily,
        "train_rets_daily": train_rets_daily,
        "bench_series_daily": bench_series_daily,
        "features_slice": features_slice,
        "trial": trial,
        "optimizer_defaults": optimizer_defaults,
    }


def test_build_objective_single_metric(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "strategy_params": {"max_lookback": 50, "leverage": 1.0},
        "optimization_metric": "Sharpe",
        "optimize": [{"parameter": "max_lookback"}, {"parameter": "leverage"}],
    }

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result = {"Sharpe": 1.5, "Calmar": 0.8}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics,
        ),
    ):
        result = objective_fn(trial)

    trial.suggest_int.assert_called_with("max_lookback", 20, 252, step=10)
    # Updated to reflect that step for float is derived from defaults/spec and can be None
    trial.suggest_float.assert_called_with("leverage", 0.5, 2.0, step=mocks["optimizer_defaults"]["leverage"]["step"], log=False)
    mock_calculate_metrics.assert_called_once()
    assert result == 1.5


def test_build_objective_multi_metric(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "strategy_params": {"param1": 10},
        "optimization_targets": [
            {"name": "Total Return", "direction": "maximize"},
            {"name": "Max Drawdown", "direction": "minimize"},
        ],
        "optimize": [
            {"parameter": "param1"}
        ],
    }
    current_optimizer_defaults = mocks["optimizer_defaults"].copy()
    current_optimizer_defaults["param1"] = {"type": "int", "low": 1, "high": 20, "step": 1}

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", current_optimizer_defaults):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result = {"Total Return": 0.25, "Max Drawdown": -0.1, "Sharpe": 1.2}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics,
        ),
    ):
        result = objective_fn(trial)

    trial.suggest_int.assert_called_with("param1", 1, 20, step=1)
    mock_calculate_metrics.assert_called_once()
    assert isinstance(result, tuple)
    assert result == (0.25, -0.1)


def test_build_objective_default_metric_when_none_specified(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "strategy_params": {"leverage": 1.0},
        "optimize": [{"parameter": "leverage"}],
    }
    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result = {
        "Calmar": 0.7,
        "Sharpe": 1.1,
    }
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics,
        ),
    ):
        result = objective_fn(trial)

    mock_calculate_metrics.assert_called_once()
    assert result == 0.7


def test_build_objective_single_metric_invalid_value(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "optimization_metric": "Sharpe",
        "strategy_params": {},
        "optimize": [],
    }
    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result_nan = {"Sharpe": np.nan}
    mock_calculate_metrics_nan = MagicMock(return_value=mock_metrics_result_nan)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics_nan,
        ),
    ):
        result_nan = objective_fn(trial)
    assert result_nan == float("-inf")

    mock_metrics_result_inf = {"Sharpe": np.inf}
    mock_calculate_metrics_inf = MagicMock(return_value=mock_metrics_result_inf)
    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics_inf,
        ),
    ):
        result_inf = objective_fn(trial)
    assert result_inf == float("-inf")


def test_build_objective_multi_metric_invalid_values(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "optimization_targets": [
            {"name": "Total Return", "direction": "maximize"},
            {"name": "Max Drawdown", "direction": "minimize"},
            {"name": "Sharpe", "direction": "maximize"},
        ],
        "strategy_params": {},
        "optimize": [],
    }
    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result = {
        "Total Return": np.nan,
        "Max Drawdown": 0.1,
        "Sharpe": np.inf,
    }
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics,
        ),
    ):
        result = objective_fn(trial)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert np.isnan(result[0])
    assert result[1] == 0.1
    assert np.isnan(result[2])


def test_build_objective_param_not_in_optimizer_config_and_no_type_in_spec(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "strategy_params": {},
        "optimization_metric": "Sharpe",
        "optimize": [
            {"parameter": "unknown_param_no_type"},
            {"parameter": "leverage"}
        ],
    }

    mock_metrics_result = {"Sharpe": 1.0}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]), \
         patch("src.portfolio_backtester.optimization.optuna_objective.print") as mock_print, \
         patch("src.portfolio_backtester.optimization.optuna_objective._run_scenario_static", mock_run_scenario), \
         patch("src.portfolio_backtester.optimization.optuna_objective.calculate_metrics", mock_calculate_metrics):

        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )
        result = objective_fn(trial)

    mock_print.assert_any_call(
        "Warning: Parameter 'unknown_param_no_type' has no type defined in scenario specification's 'optimize' section or in OPTIMIZER_PARAMETER_DEFAULTS. Skipping parameter."
    )
    trial.suggest_float.assert_called_with("leverage", 0.5, 2.0, step=mocks["optimizer_defaults"]["leverage"]["step"], log=False)
    mock_calculate_metrics.assert_called_once()
    assert result == 1.0


def test_build_objective_with_constraint_violation(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "optimization_metric": "Total Return",
        "optimization_constraints": [{"metric": "Max Drawdown", "min_value": -0.1}],
        "strategy_params": {},
        "optimize": [],
    }

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result = {"Total Return": 0.2, "Max Drawdown": -0.2}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics,
        ),
    ):
        result = objective_fn(trial)

    assert result == float("-inf")


def test_build_objective_with_constraint_satisfied(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]

    base_scen_cfg = {
        "optimization_metric": "Total Return",
        "optimization_constraints": [{"metric": "Max Drawdown", "min_value": -0.1}],
        "strategy_params": {},
        "optimize": [],
    }

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"],
            base_scen_cfg,
            mocks["train_data_monthly"],
            mocks["train_data_daily"],
            mocks["train_rets_daily"],
            mocks["bench_series_daily"],
            mocks["features_slice"],
        )

    mock_metrics_result = {"Total Return": 0.3, "Max Drawdown": -0.05}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)
    mock_run_scenario = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.portfolio_backtester.optimization.optuna_objective._run_scenario_static",
            mock_run_scenario,
        ),
        patch(
            "src.portfolio_backtester.optimization.optuna_objective.calculate_metrics",
            mock_calculate_metrics,
        ),
    ):
        result = objective_fn(trial)

    assert result == 0.3


# --- New tests for categorical parameters and overrides ---

def test_suggest_categorical_numeric(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]
    base_scen_cfg = {
        "strategy_params": {},
        "optimize": [{"parameter": "numeric_enum"}],
        "optimization_metric": "Sharpe",
    }
    trial.suggest_categorical.return_value = 20 # Mock Optuna's suggestion

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"], base_scen_cfg, mocks["train_data_monthly"], mocks["train_data_daily"],
            mocks["train_rets_daily"], mocks["bench_series_daily"], mocks["features_slice"]
        )

    mock_run_scenario = MagicMock(return_value=MagicMock())
    with patch("src.portfolio_backtester.optimization.optuna_objective._run_scenario_static", mock_run_scenario), \
         patch("src.portfolio_backtester.optimization.optuna_objective.calculate_metrics", return_value={"Sharpe": 1.0}):
        objective_fn(trial)

    trial.suggest_categorical.assert_called_with("numeric_enum", [10, 20, 30])
    # Check that the suggested value is passed to _run_scenario_static via strategy_params
    args, kwargs = mock_run_scenario.call_args
    scen_cfg_passed = args[1] # scen_cfg is the second argument to _run_scenario_static
    assert scen_cfg_passed["strategy_params"]["numeric_enum"] == 20


def test_suggest_categorical_string_for_special_key(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]
    base_scen_cfg = {
        "strategy_params": {},
        "optimize": [{"parameter": "position_sizer"}], # This is a SPECIAL_SCEN_CFG_KEYS
        "optimization_metric": "Sharpe",
        "position_sizer": "default_sizer" # Initial value in base_scen_cfg
    }
    trial.suggest_categorical.return_value = "rolling_sharpe"

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"], base_scen_cfg, mocks["train_data_monthly"], mocks["train_data_daily"],
            mocks["train_rets_daily"], mocks["bench_series_daily"], mocks["features_slice"]
        )

    mock_run_scenario = MagicMock(return_value=MagicMock())
    with patch("src.portfolio_backtester.optimization.optuna_objective._run_scenario_static", mock_run_scenario), \
         patch("src.portfolio_backtester.optimization.optuna_objective.calculate_metrics", return_value={"Sharpe": 1.0}):
        objective_fn(trial)

    trial.suggest_categorical.assert_called_with("position_sizer", ["equal_weight", "rolling_sharpe"])
    args, kwargs = mock_run_scenario.call_args
    scen_cfg_passed = args[1]
    assert scen_cfg_passed["position_sizer"] == "rolling_sharpe" # Check direct scen_cfg override
    assert "position_sizer" not in scen_cfg_passed["strategy_params"] # Should not be in strategy_params


def test_override_param_type_from_int_to_categorical(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]
    base_scen_cfg = {
        "strategy_params": {},
        "optimize": [{
            "parameter": "param_to_override",
            "type": "categorical", # Override type
            "values": ["cat1", "cat2"]     # Provide choices for new type
        }],
        "optimization_metric": "Sharpe",
    }
    # param_to_override is "int" in mocks["optimizer_defaults"]
    trial.suggest_categorical.return_value = "cat1"

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"], base_scen_cfg, mocks["train_data_monthly"], mocks["train_data_daily"],
            mocks["train_rets_daily"], mocks["bench_series_daily"], mocks["features_slice"]
        )

    mock_run_scenario = MagicMock(return_value=MagicMock())
    with patch("src.portfolio_backtester.optimization.optuna_objective._run_scenario_static", mock_run_scenario), \
         patch("src.portfolio_backtester.optimization.optuna_objective.calculate_metrics", return_value={"Sharpe": 1.0}):
        objective_fn(trial)

    trial.suggest_categorical.assert_called_with("param_to_override", ["cat1", "cat2"])
    assert not trial.suggest_int.called # Ensure suggest_int was NOT called
    args, kwargs = mock_run_scenario.call_args
    scen_cfg_passed = args[1]
    assert scen_cfg_passed["strategy_params"]["param_to_override"] == "cat1"


def test_override_categorical_choices_from_scenario(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]
    base_scen_cfg = {
        "strategy_params": {},
        "optimize": [{
            "parameter": "default_choices_param",
            "type": "categorical", # Can optionally repeat type
            "values": ["scenario_val1", "scenario_val2"] # Override choices
        }],
        "optimization_metric": "Sharpe",
    }
    # default_choices_param has ["default1", "default2"] in mocks["optimizer_defaults"]
    trial.suggest_categorical.return_value = "scenario_val1"

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"], base_scen_cfg, mocks["train_data_monthly"], mocks["train_data_daily"],
            mocks["train_rets_daily"], mocks["bench_series_daily"], mocks["features_slice"]
        )

    mock_run_scenario = MagicMock(return_value=MagicMock())
    with patch("src.portfolio_backtester.optimization.optuna_objective._run_scenario_static", mock_run_scenario), \
         patch("src.portfolio_backtester.optimization.optuna_objective.calculate_metrics", return_value={"Sharpe": 1.0}):
        objective_fn(trial)

    trial.suggest_categorical.assert_called_with("default_choices_param", ["scenario_val1", "scenario_val2"])
    args, kwargs = mock_run_scenario.call_args
    scen_cfg_passed = args[1]
    assert scen_cfg_passed["strategy_params"]["default_choices_param"] == "scenario_val1"


def test_error_categorical_no_choices(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]
    base_scen_cfg = {
        "strategy_params": {},
        "optimize": [{"parameter": "missing_choices_param", "type": "categorical"}], # No values in spec
        "optimization_metric": "Sharpe",
    }
    # missing_choices_param is NOT in mocks["optimizer_defaults"]

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"], base_scen_cfg, mocks["train_data_monthly"], mocks["train_data_daily"],
            mocks["train_rets_daily"], mocks["bench_series_daily"], mocks["features_slice"]
        )

    with pytest.raises(ValueError) as excinfo:
        objective_fn(trial)
    assert "missing_choices_param' has no choices defined or choices are invalid" in str(excinfo.value)


def test_error_unsupported_param_type(common_mocks):
    mocks = common_mocks
    trial = mocks["trial"]
    base_scen_cfg = {
        "strategy_params": {},
        "optimize": [{"parameter": "bad_type_param", "type": "unsupported_type"}],
        "optimization_metric": "Sharpe",
    }

    with patch("src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS", mocks["optimizer_defaults"]):
        objective_fn = build_objective(
            mocks["g_cfg"], base_scen_cfg, mocks["train_data_monthly"], mocks["train_data_daily"],
            mocks["train_rets_daily"], mocks["bench_series_daily"], mocks["features_slice"]
        )

    with pytest.raises(ValueError) as excinfo:
        objective_fn(trial)
    assert "Unsupported parameter type 'unsupported_type' for parameter 'bad_type_param'" in str(excinfo.value)
