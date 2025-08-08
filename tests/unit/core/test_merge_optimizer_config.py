from portfolio_backtester.config_loader import merge_optimizer_config


def test_merge_optimizer_config_adds_keys_and_removes_section():
    scenario = {
        "optimizers": {
            "optuna": {
                "optimize": [
                    {"parameter": "p", "min_value": 1, "max_value": 2}
                ],
                "foo": 123,
            }
        }
    }
    merged = merge_optimizer_config(scenario, "optuna")
    # Old mapping removed
    assert "optimizers" not in merged
    # Keys promoted
    assert merged["optimize"][0]["parameter"] == "p"
    assert merged["foo"] == 123


def test_merge_optimizer_config_deep_merge():
    scenario = {
        "existing": {"a": 1},
        "optimizers": {
            "optuna": {
                "existing": {"b": 2},
                "new": 3,
            }
        }
    }
    merged = merge_optimizer_config(scenario, "optuna")
    # existing dict should be updated with nested values
    assert merged["existing"] == {"a": 1, "b": 2}
    # new field added
    assert merged["new"] == 3
