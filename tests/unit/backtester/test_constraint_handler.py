import pytest

from portfolio_backtester.backtester_logic.constraint_handler import ConstraintHandler


def _build_handler():
    return ConstraintHandler(global_config={})


@pytest.mark.parametrize(
    "current_leverage, attempt, expected_leverage",
    [
        (2.0, 0, 2.0 * 0.7),  # first attempt reduces by 30 %
        (1.4, 1, 1.4 * 0.7**2),  # second attempt continues reduction
    ],
)
def test_adjust_parameters_reduces_leverage(current_leverage, attempt, expected_leverage):
    handler = _build_handler()
    params = {"leverage": current_leverage}
    violations = ["Ann. Vol = 0.30 > 0.20 (max)"]
    scenario_config = {"min_leverage_allowed": 0.1}

    adjusted = handler._adjust_parameters_for_constraints(
        params, violations, scenario_config, attempt
    )

    # Leverage should be reduced but not increase
    assert pytest.approx(expected_leverage, rel=1e-6) == adjusted["leverage"]


def test_adjust_parameters_respects_min_leverage():
    handler = _build_handler()
    params = {"leverage": 0.11}  # already near min
    violations = ["Ann. Vol = 0.25 > 0.2 (max)"]
    scenario_config = {"min_leverage_allowed": 0.1}

    # Reduction would go below min; expect leverage clamp to min_leverage_allowed
    adjusted = handler._adjust_parameters_for_constraints(
        params, violations, scenario_config, attempt=0
    )
    assert adjusted["leverage"] == pytest.approx(0.1)


def test_check_constraint_violations_detects_bounds():
    handler = _build_handler()
    metrics = {"Sharpe": 0.5, "Ann. Vol": 0.25}
    constraints = [
        {"metric": "Sharpe", "min_value": 0.8},
        {"metric": "Ann. Vol", "max_value": 0.2},
    ]

    vios = handler._check_constraint_violations(metrics, constraints)

    # Both constraints violated
    assert len(vios) == 2
    assert any("Sharpe" in v for v in vios)
    assert any("Ann. Vol" in v for v in vios)
