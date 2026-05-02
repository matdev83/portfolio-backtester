"""Tests for research protocol configuration parsing and validation."""

from __future__ import annotations

import pytest

from portfolio_backtester.research.protocol_config import (
    ROBUST_COMPOSITE_METRIC_NAME,
    CostSensitivityRunOn,
    DoubleOOSWFOProtocolConfig,
    ExecutionConfig,
    FinalUnseenMode,
    ResearchProtocolConfigError,
    ResumePartialRunConfig,
    WFOGridConfig,
    parse_double_oos_wfo_protocol,
)


def _minimal_primary_inner() -> dict:
    return {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train_period": {
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
        },
        "unseen_test_period": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        },
        "wfo_window_grid": {
            "train_window_months": [24],
            "test_window_months": [6],
            "wfo_step_months": [3],
            "walk_forward_type": ["rolling"],
        },
        "selection": {"top_n": 3, "metric": "Calmar"},
        "final_unseen_mode": "fixed_selected_params",
        "lock": {"enabled": True, "refuse_overwrite": False},
        "reporting": {"enabled": True},
    }


def _minimal_alias_inner() -> dict:
    return {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train": {"start": "2020-01-01", "end": "2022-12-31"},
        "unseen_holdout": {"start": "2023-01-01", "end": "2023-12-31"},
        "wfo_grid": {
            "train_months": [24],
            "test_months": [6],
            "step_months": [3],
            "walk_forward_types": ["rolling"],
        },
        "selection": {"top_n": 3, "metric": "Sharpe"},
    }


def test_execution_defaults_max_grid_cells_and_fail_fast() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    assert cfg.execution == ExecutionConfig(
        max_grid_cells=100,
        fail_fast=True,
        max_parallel_grid_workers=1,
        resume_partial=ResumePartialRunConfig(),
    )


def test_execution_parse_partial_overrides_defaults() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {"max_grid_cells": 50}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.execution.max_grid_cells == 50
    assert cfg.execution.fail_fast is True

    inner["execution"] = {"fail_fast": False}
    cfg2 = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg2.execution.max_grid_cells == 100
    assert cfg2.execution.fail_fast is False


def test_execution_non_mapping_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = "no"
    with pytest.raises(ResearchProtocolConfigError, match="execution"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_execution_max_grid_cells_non_positive_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {"max_grid_cells": 0}
    with pytest.raises(ResearchProtocolConfigError, match="max_grid_cells"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_execution_max_grid_cells_non_int_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {"max_grid_cells": "100"}
    with pytest.raises(ResearchProtocolConfigError, match="integer"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_execution_max_parallel_grid_workers_below_one_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {"max_parallel_grid_workers": 0}
    with pytest.raises(ResearchProtocolConfigError, match="max_parallel_grid_workers"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_execution_max_parallel_grid_workers_non_int_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {"max_parallel_grid_workers": "2"}
    with pytest.raises(ResearchProtocolConfigError, match="integer"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_cross_validation_defaults_when_disabled_allow_n_folds_one() -> None:
    inner = _minimal_primary_inner()
    inner["cross_validation"] = {"enabled": False, "n_folds": 1}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.cross_validation.enabled is False
    assert cfg.cross_validation.n_folds == 1


def test_constraints_defaults_empty_tuple() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    assert cfg.constraints == ()


def test_parse_primary_yaml_names_double_oos_wfo() -> None:
    raw = {"research_protocol": _minimal_primary_inner()}
    cfg = parse_double_oos_wfo_protocol(raw)
    assert isinstance(cfg, DoubleOOSWFOProtocolConfig)
    assert cfg.enabled is True
    assert cfg.selection.metric == "Calmar"
    assert cfg.selection.top_n == 3
    assert cfg.final_unseen_mode == FinalUnseenMode.FIXED_SELECTED_PARAMS
    assert cfg.lock.refuse_overwrite is False


def test_parsed_config_includes_random_strategy_parameters_bootstrap() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 50,
        "random_strategy_parameters": {"enabled": True, "sample_size": 120},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    rsp = cfg.bootstrap.random_strategy_parameters
    assert rsp.enabled is True
    assert rsp.sample_size == 120


def test_global_train_grid_aliases_still_parse() -> None:
    raw = {"research_protocol": _minimal_alias_inner()}
    cfg = parse_double_oos_wfo_protocol(raw)
    assert cfg.wfo_window_grid.train_window_months == (24,)
    assert cfg.wfo_window_grid.test_window_months == (6,)
    assert cfg.wfo_window_grid.wfo_step_months == (3,)
    assert cfg.wfo_window_grid.walk_forward_type == ("rolling",)


def test_architecture_lock_alias_under_lock_keys() -> None:
    inner = _minimal_primary_inner().copy()
    del inner["lock"]
    inner["architecture_lock"] = {"enabled": False, "refuse_overwrite": True}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.lock.enabled is False
    assert cfg.lock.refuse_overwrite is True


def test_walk_forward_type_scalar_normalized() -> None:
    inner = _minimal_primary_inner()
    inner = {
        **inner,
        "wfo_window_grid": {**inner["wfo_window_grid"], "walk_forward_type": "expanding"},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.wfo_window_grid.walk_forward_type == ("expanding",)


def test_missing_research_protocol_raises() -> None:
    with pytest.raises(ResearchProtocolConfigError, match="research_protocol"):
        parse_double_oos_wfo_protocol({})


def test_disabled_protocol_parse_then_validate_raises() -> None:
    inner = _minimal_primary_inner()
    inner["enabled"] = False
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.enabled is False
    with pytest.raises(ResearchProtocolConfigError, match="disabled"):
        cfg.validate_enabled()
    with pytest.raises(ResearchProtocolConfigError, match="disabled"):
        cfg.research_validate()
    with pytest.raises(ResearchProtocolConfigError, match="disabled"):
        cfg.validate_for_mode("optimize")


def test_unknown_type_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["type"] = "unknown_proto"
    with pytest.raises(ResearchProtocolConfigError, match="type"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_overlapping_train_unseen_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["unseen_test_period"] = {"start_date": "2022-06-01", "end_date": "2024-01-01"}
    with pytest.raises(ResearchProtocolConfigError, match="overlap"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_empty_grid_train_window_months_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"]["train_window_months"] = []
    with pytest.raises(ResearchProtocolConfigError, match="train_window_months"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_non_positive_grid_value_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"]["test_window_months"] = [0, 6]
    with pytest.raises(ResearchProtocolConfigError, match="positive"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_unknown_walk_forward_type_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"]["walk_forward_type"] = ["rolling", "bogus"]
    with pytest.raises(ResearchProtocolConfigError, match="walk_forward"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_top_n_non_positive_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["selection"]["top_n"] = 0
    with pytest.raises(ResearchProtocolConfigError, match="top_n"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_final_unseen_mode_unknown_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["final_unseen_mode"] = "not_a_mode"
    with pytest.raises(ResearchProtocolConfigError, match="final_unseen"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_defaults_final_unseen_lock_reporting() -> None:
    inner = _minimal_primary_inner()
    del inner["final_unseen_mode"]
    del inner["lock"]
    del inner["reporting"]
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.final_unseen_mode == FinalUnseenMode.REOPTIMIZE_WITH_LOCKED_ARCHITECTURE
    assert cfg.lock.enabled is True
    assert cfg.lock.refuse_overwrite is True
    assert cfg.reporting.enabled is True
    assert cfg.reporting.generate_heatmaps is False
    assert cfg.reporting.generate_html is False
    assert cfg.reporting.heatmap_metrics == ("score", "robust_score")
    assert cfg.reporting.html_embed_figures is False
    assert cfg.reporting.html_navigation is True
    assert cfg.reporting.generate_bootstrap_distribution_plots is False
    assert cfg.reporting.generate_cost_sensitivity_figure is False


def test_reporting_generate_html_parse() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {"enabled": True, "generate_html": True}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.reporting.generate_html is True


def test_reporting_html_optional_flags_parse() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {"enabled": True, "n_samples": 20}
    inner["cost_sensitivity"] = {
        "enabled": True,
        "slippage_bps_grid": [0, 5],
        "commission_multiplier_grid": [1.0],
        "run_on": "unseen",
    }
    inner["reporting"] = {
        "enabled": True,
        "generate_html": True,
        "html_embed_figures": True,
        "html_navigation": False,
        "generate_bootstrap_distribution_plots": True,
        "generate_cost_sensitivity_figure": True,
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.reporting.generate_html is True
    assert cfg.reporting.html_embed_figures is True
    assert cfg.reporting.html_navigation is False
    assert cfg.reporting.generate_bootstrap_distribution_plots is True
    assert cfg.reporting.generate_cost_sensitivity_figure is True


def test_reporting_generate_heatmaps_and_metrics_parse() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {
        "enabled": True,
        "generate_heatmaps": True,
        "heatmap_metrics": ["score", "robust_score", "Calmar", "Max Drawdown"],
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.reporting.generate_heatmaps is True
    assert cfg.reporting.heatmap_metrics == (
        "score",
        "robust_score",
        "Calmar",
        "Max Drawdown",
    )


def test_reporting_heatmap_metrics_default_when_generate_true() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {"enabled": True, "generate_heatmaps": True}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.reporting.heatmap_metrics == ("score", "robust_score")


def test_reporting_heatmap_empty_metrics_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {
        "enabled": True,
        "generate_heatmaps": True,
        "heatmap_metrics": [],
    }
    with pytest.raises(ResearchProtocolConfigError, match="heatmap_metrics"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_reporting_heatmap_unknown_metric_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {
        "enabled": True,
        "generate_heatmaps": True,
        "heatmap_metrics": ["not_a_real_metric_xyz"],
    }
    with pytest.raises(ResearchProtocolConfigError, match="heatmap"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_wfo_grid_config_dataclass_public_field_names() -> None:
    g = WFOGridConfig(
        train_window_months=(12, 24),
        test_window_months=(6,),
        wfo_step_months=(3,),
        walk_forward_type=("rolling", "expanding"),
    )
    assert g.train_window_months == (12, 24)
    assert g.test_window_months == (6,)
    assert g.wfo_step_months == (3,)
    assert g.walk_forward_type == ("rolling", "expanding")


def test_selection_metric_omitted_defaults_robust_composite_and_scoring() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.selection.metric == ROBUST_COMPOSITE_METRIC_NAME
    assert cfg.composite_scoring is not None
    assert cfg.composite_scoring.weights[0][0] == "Calmar"
    assert cfg.composite_scoring.directions["Turnover"] == "lower"
    assert cfg.composite_scoring.directions["Max Drawdown"] == "higher"


def test_selection_metric_robust_composite_case_insensitive() -> None:
    inner = _minimal_primary_inner()
    inner["selection"]["metric"] = "robustcomposite"
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.selection.metric == ROBUST_COMPOSITE_METRIC_NAME


def test_scoring_composite_rank_custom_weights_and_directions() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    inner["scoring"] = {
        "type": "composite_rank",
        "weights": {
            "Calmar": 0.35,
            "Sortino": 0.25,
            "Total Return": 0.20,
            "Max Drawdown": 0.10,
            "Turnover": 0.10,
        },
        "directions": {"Turnover": "lower"},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.composite_scoring is not None
    wmap = dict(cfg.composite_scoring.weights)
    assert wmap["Calmar"] == pytest.approx(0.35)
    assert cfg.composite_scoring.directions["Turnover"] == "lower"


def test_scoring_rejected_when_metric_is_single_metric() -> None:
    inner = _minimal_primary_inner()
    inner["scoring"] = {"type": "composite_rank", "weights": {"Calmar": 1.0}}
    with pytest.raises(ResearchProtocolConfigError, match="scoring"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_scoring_empty_weights_rejected() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    inner["scoring"] = {"type": "composite_rank", "weights": {}}
    with pytest.raises(ResearchProtocolConfigError, match="weights"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_scoring_negative_weight_rejected() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    inner["scoring"] = {"type": "composite_rank", "weights": {"Calmar": -0.5, "Sortino": 1.5}}
    with pytest.raises(ResearchProtocolConfigError, match="weight"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_scoring_unknown_direction_rejected() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    inner["scoring"] = {
        "type": "composite_rank",
        "weights": {"Calmar": 1.0},
        "directions": {"Calmar": "sideways"},
    }
    with pytest.raises(ResearchProtocolConfigError, match="direction"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_scoring_zero_weight_entry_rejected() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    inner["scoring"] = {"type": "composite_rank", "weights": {"Calmar": 0.0, "Sortino": 1.0}}
    with pytest.raises(ResearchProtocolConfigError, match="weight"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_scoring_unsupported_type_rejected() -> None:
    inner = _minimal_primary_inner()
    del inner["selection"]["metric"]
    inner["scoring"] = {"type": "other", "weights": {"Calmar": 1.0}}
    with pytest.raises(ResearchProtocolConfigError, match="type"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_robust_selection_defaults_disabled() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    assert cfg.robust_selection.enabled is False
    assert cfg.robust_selection.cell_weight == pytest.approx(0.5)
    assert cfg.robust_selection.neighbor_median_weight == pytest.approx(0.3)
    assert cfg.robust_selection.neighbor_min_weight == pytest.approx(0.2)


def test_robust_selection_parse_enabled_and_weights() -> None:
    inner = _minimal_primary_inner()
    inner["robust_selection"] = {
        "enabled": True,
        "weights": {"cell": 0.50, "neighbor_median": 0.30, "neighbor_min": 0.20},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.robust_selection.enabled is True
    assert cfg.robust_selection.cell_weight == pytest.approx(0.5)


def test_robust_selection_non_mapping_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["robust_selection"] = "no"
    with pytest.raises(ResearchProtocolConfigError, match="robust_selection"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_robust_selection_weight_non_positive_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["robust_selection"] = {
        "enabled": True,
        "weights": {"cell": 0.0, "neighbor_median": 1.0, "neighbor_min": 1.0},
    }
    with pytest.raises(ResearchProtocolConfigError, match="weight"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_cost_sensitivity_defaults_disabled_and_unseen() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    assert cfg.cost_sensitivity.enabled is False
    assert cfg.cost_sensitivity.run_on == CostSensitivityRunOn.UNSEEN
    assert cfg.cost_sensitivity.slippage_bps_grid == ()
    assert cfg.cost_sensitivity.commission_multiplier_grid == (1.0,)


def test_cost_sensitivity_parse_grid_and_enabled() -> None:
    inner = _minimal_primary_inner()
    inner["cost_sensitivity"] = {
        "enabled": True,
        "slippage_bps_grid": [0, 2.5, 5],
        "commission_multiplier_grid": [1.0, 2.0],
        "run_on": "unseen",
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.cost_sensitivity.enabled is True
    assert cfg.cost_sensitivity.slippage_bps_grid == (0.0, 2.5, 5.0)
    assert cfg.cost_sensitivity.commission_multiplier_grid == (1.0, 2.0)


def test_cost_sensitivity_unknown_run_on_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["cost_sensitivity"] = {
        "enabled": True,
        "slippage_bps_grid": [0.0],
        "commission_multiplier_grid": [1.0],
        "run_on": "global_train",
    }
    with pytest.raises(ResearchProtocolConfigError, match="run_on"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_cost_sensitivity_enabled_requires_nonempty_grids() -> None:
    inner = _minimal_primary_inner()
    inner["cost_sensitivity"] = {
        "enabled": True,
        "slippage_bps_grid": [],
        "commission_multiplier_grid": [1.0],
    }
    with pytest.raises(ResearchProtocolConfigError, match="slippage"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_robust_selection_weights_missing_key_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["robust_selection"] = {
        "enabled": True,
        "weights": {"cell": 1.0, "neighbor_median": 1.0},
    }
    with pytest.raises(ResearchProtocolConfigError, match="neighbor_min"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_subblocks_default_when_parent_enabled() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {"enabled": True, "n_samples": 50, "random_seed": 1}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.bootstrap.enabled is True
    assert cfg.bootstrap.n_samples == 50
    assert cfg.bootstrap.random_seed == 1
    assert cfg.bootstrap.random_wfo_architecture.enabled is False
    assert cfg.bootstrap.block_shuffled_returns.enabled is False
    assert cfg.bootstrap.block_shuffled_positions.enabled is False
    assert cfg.bootstrap.random_strategy_parameters.enabled is False
    assert cfg.bootstrap.persist_distribution_samples is False


def test_bootstrap_persist_distribution_samples_parse() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 50,
        "random_seed": 1,
        "persist_distribution_samples": True,
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.bootstrap.persist_distribution_samples is True


def test_bootstrap_block_shuffled_positions_subblock_non_mapping_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 10,
        "block_shuffled_positions": "yes",
    }
    with pytest.raises(ResearchProtocolConfigError, match="block_shuffled_positions"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_random_wfo_subblock_non_mapping_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 10,
        "random_wfo_architecture": "yes",
    }
    with pytest.raises(ResearchProtocolConfigError, match="random_wfo"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_random_strategy_parameters_subblock_non_mapping_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 10,
        "random_strategy_parameters": "yes",
    }
    with pytest.raises(ResearchProtocolConfigError, match="random_strategy_parameters"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_reporting_bootstrap_plots_require_bootstrap_enabled() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {
        "enabled": True,
        "generate_html": True,
        "generate_bootstrap_distribution_plots": True,
    }
    inner["bootstrap"] = {"enabled": False, "n_samples": 10}
    with pytest.raises(ResearchProtocolConfigError, match="bootstrap"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_reporting_cost_sensitivity_figure_requires_cost_sensitivity_enabled() -> None:
    inner = _minimal_primary_inner()
    inner["reporting"] = {
        "enabled": True,
        "generate_html": True,
        "generate_cost_sensitivity_figure": True,
    }
    inner["cost_sensitivity"] = {"enabled": False}
    with pytest.raises(ResearchProtocolConfigError, match="cost_sensitivity"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_cross_validation_unknown_strategy_rejected_when_disabled() -> None:
    inner = _minimal_primary_inner()
    inner["cross_validation"] = {"enabled": False, "strategy": "kfold_split"}
    with pytest.raises(ResearchProtocolConfigError, match="cross_validation.strategy"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})
