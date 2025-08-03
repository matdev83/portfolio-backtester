import os
import shutil
import tempfile
from pathlib import Path

import pytest

from portfolio_backtester.strategy_config_validator import validate_strategy_configs


@pytest.fixture
def temp_project_structure(tmp_path: Path):
    """Creates a temporary directory structure mimicking the project layout."""
    # Create temporary src and config directories
    src_dir = tmp_path / "src" / "portfolio_backtester" / "strategies"
    scenarios_dir = tmp_path / "config" / "scenarios"
    src_dir.mkdir(parents=True)
    scenarios_dir.mkdir(parents=True)

    # Strategy categories
    categories = ["signal", "portfolio", "diagnostic", "meta"]
    for cat in categories:
        (src_dir / cat).mkdir()
        (scenarios_dir / cat).mkdir()

    # Base directory (should be skipped)
    (src_dir / "base").mkdir()

    # Utility files in root strategies dir (should be skipped)
    (src_dir / "strategy_factory.py").touch()
    (src_dir / "candidate_weights.py").touch()
    (src_dir / "leverage_and_smoothing.py").touch()


    # --- Test Case 1: Well-formed strategy 'good_signal_strategy' ---
    good_signal_strategy_dir = src_dir / "signal" / "good_signal_strategy"
    good_signal_strategy_dir.mkdir()
    (good_signal_strategy_dir / "good_signal_strategy.py").touch()
    (good_signal_strategy_dir / "helper_utils.py").touch() # This one will be warned about
    (good_signal_strategy_dir / "__init__.py").touch()
    # Corresponding config
    good_config_dir = scenarios_dir / "signal" / "good_signal_strategy"
    good_config_dir.mkdir()
    with open(good_config_dir / "default.yaml", "w") as f:
        f.write("name: good_signal_strategy_test\nstrategy: good_signal_strategy\nparam1: 10")

    # --- Test Case 2: Strategy 'missing_config_strategy' without default.yaml ---
    missing_config_strategy_dir = src_dir / "portfolio" / "missing_config_strategy"
    missing_config_strategy_dir.mkdir()
    (missing_config_strategy_dir / "missing_config_strategy.py").touch()
    # No corresponding config directory or default.yaml

    # --- Test Case 3: Strategy 'bad_py_naming_strategy' with a poorly named .py file ---
    bad_py_naming_strategy_dir = src_dir / "diagnostic" / "bad_py_naming_strategy"
    bad_py_naming_strategy_dir.mkdir()
    (bad_py_naming_strategy_dir / "bad_py_naming_strategy.py").touch()
    (bad_py_naming_strategy_dir / "some_random_helper.py").touch() # This one will be warned about
    # Corresponding config
    bad_config_dir = scenarios_dir / "diagnostic" / "bad_py_naming_strategy"
    bad_config_dir.mkdir()
    with open(bad_config_dir / "default.yaml", "w") as f:
        f.write("name: bad_py_naming_strategy_test\nstrategy: bad_py_naming_strategy\n")

    # --- Test Case 4: Strategy 'empty_strategy_dir' ---
    empty_strategy_dir = src_dir / "meta" / "empty_strategy_dir"
    empty_strategy_dir.mkdir()
    # No .py files, no config
    # Corresponding config (should cause an error for missing config)
    empty_config_dir = scenarios_dir / "meta" / "empty_strategy_dir"
    empty_config_dir.mkdir()
    with open(empty_config_dir / "default.yaml", "w") as f: # Config exists, but no .py files to check naming
        f.write("name: empty_strategy_dir_test\nstrategy: empty_strategy_dir\n")


    # --- Test Case 5: Strategy 'all_good_py_naming' with all .py files correctly named ---
    all_good_py_naming_dir = src_dir / "signal" / "all_good_py_naming"
    all_good_py_naming_dir.mkdir()
    (all_good_py_naming_dir / "all_good_py_naming.py").touch()
    (all_good_py_naming_dir / "all_good_py_naming_sub_module.py").touch()
    (all_good_py_naming_dir / "__init__.py").touch()
    # Corresponding config
    all_good_config_dir = scenarios_dir / "signal" / "all_good_py_naming"
    all_good_config_dir.mkdir()
    with open(all_good_config_dir / "default.yaml", "w") as f:
        f.write("name: all_good_py_naming_test\nstrategy: all_good_py_naming\n")

    # --- Test Case 6: File in unknown category (should be skipped by validation) ---
    unknown_category_dir = src_dir / "unknown_category"
    unknown_category_dir.mkdir()
    (unknown_category_dir / "unknown_strategy.py").touch()
    # No corresponding config

    return str(src_dir), str(scenarios_dir)


def test_validate_strategy_configs_all_good(temp_project_structure):
    """
    Test with the current fixture.
    Note: This fixture includes 'missing_config_strategy', 'helper_utils.py', and 'some_random_helper.py'.
    So, it's NOT "all good" in an absolute sense.
    """
    src_dir, scenarios_dir = temp_project_structure
    is_valid, errors = validate_strategy_configs(src_dir, scenarios_dir)
    assert not is_valid # Should be invalid due to missing_config_strategy and bad .py naming
    assert len(errors) == 3 # missing_config_strategy, helper_utils.py, some_random_helper.py
    assert any("missing_config_strategy' does not have a corresponding default.yaml" in error for error in errors)
    assert "helper_utils.py' in strategy 'good_signal_strategy' directory does not contain the strategy name" in "\n".join(errors)
    assert "some_random_helper.py' in strategy 'bad_py_naming_strategy' directory does not contain the strategy name" in "\n".join(errors)


def test_validate_strategy_configs_missing_default_yaml(temp_project_structure):
    """Test that missing default.yaml is reported as an error (among other expected errors)."""
    src_dir, scenarios_dir = temp_project_structure
    is_valid, errors = validate_strategy_configs(src_dir, scenarios_dir)
    assert not is_valid
    assert any("missing_config_strategy' does not have a corresponding default.yaml" in error for error in errors)
    assert len(errors) == 3 # missing_config_strategy, helper_utils.py, some_random_helper.py


def test_validate_strategy_configs_bad_py_naming(temp_project_structure):
    """Test that poorly named .py files are reported (along with other expected errors)."""
    src_dir, scenarios_dir = temp_project_structure
    is_valid, errors = validate_strategy_configs(src_dir, scenarios_dir)
    assert not is_valid

    error_messages = "\n".join(errors)
    assert "helper_utils.py' in strategy 'good_signal_strategy' directory does not contain the strategy name" in error_messages
    assert "some_random_helper.py' in strategy 'bad_py_naming_strategy' directory does not contain the strategy name" in error_messages
    assert "missing_config_strategy' does not have a corresponding default.yaml" in error_messages
    assert len(errors) == 3


def test_validate_strategy_configs_all_good_py_naming(temp_project_structure):
    """Test that 'all_good_py_naming' strategy itself doesn't produce .py naming errors (but others do)."""
    src_dir, scenarios_dir = temp_project_structure
    is_valid, errors = validate_strategy_configs(src_dir, scenarios_dir)
    assert not is_valid
    
    error_messages = "\n".join(errors)
    assert "all_good_py_naming_sub_module.py' in strategy 'all_good_py_naming' directory does not contain the strategy name ('all_good_py_naming') in its filename." not in error_messages
    assert "missing_config_strategy' does not have a corresponding default.yaml" in error_messages
    assert len(errors) == 3 # missing_config_strategy, helper_utils.py, some_random_helper.py


def test_validate_strategy_configs_empty_strategy_dir(temp_project_structure):
    """Test 'empty_strategy_dir' (it has a config, so no missing_config error for it, but others exist)."""
    src_dir, scenarios_dir = temp_project_structure
    is_valid, errors = validate_strategy_configs(src_dir, scenarios_dir)
    assert not is_valid # Still invalid due to other strategies

    error_messages = "\n".join(errors)
    assert "empty_strategy_dir' directory does not contain the strategy name" not in error_messages # No .py files to check for *this* strategy
    assert "missing_config_strategy' does not have a corresponding default.yaml" in error_messages
    assert len(errors) == 3 # missing_config_strategy, helper_utils.py, some_random_helper.py

def test_validate_strategy_configs_skips_unknown_category(temp_project_structure):
    """Test that 'unknown_category' is skipped (but other errors remain)."""
    src_dir, scenarios_dir = temp_project_structure
    is_valid, errors = validate_strategy_configs(src_dir, scenarios_dir)
    assert not is_valid

    error_messages = "\n".join(errors)
    assert "unknown_strategy' does not have a corresponding default.yaml" not in error_messages
    assert "unknown_strategy' directory does not contain the strategy name" not in error_messages
    assert "missing_config_strategy' does not have a corresponding default.yaml" in error_messages
    assert len(errors) == 3 # missing_config_strategy, helper_utils.py, some_random_helper.py

def test_validate_strategy_configs_skips_base_and_utility_files(tmp_path: Path):
    """Test that files in 'base' directory and known utility files are skipped."""
    src_dir = tmp_path / "src" / "portfolio_backtester" / "strategies"
    scenarios_dir = tmp_path / "config" / "scenarios"
    src_dir.mkdir(parents=True)
    scenarios_dir.mkdir(parents=True)

    # Create 'base' directory and a file in it
    base_dir = src_dir / "base"
    base_dir.mkdir()
    (base_dir / "base_strategy.py").touch()

    # Create utility files in root strategies dir
    (src_dir / "strategy_factory.py").touch()
    (src_dir / "candidate_weights.py").touch()
    (src_dir / "leverage_and_smoothing.py").touch()

    # Create 'signal' category directory
    (src_dir / "signal").mkdir(exist_ok=True)
    # Create 'signal' category directory in scenarios
    (scenarios_dir / "signal").mkdir(exist_ok=True)
    # Create a valid strategy
    valid_strategy_dir = src_dir / "signal" / "my_strategy"
    valid_strategy_dir.mkdir()
    (valid_strategy_dir / "my_strategy.py").touch()
    valid_config_dir = scenarios_dir / "signal" / "my_strategy"
    valid_config_dir.mkdir()
    with open(valid_config_dir / "default.yaml", "w") as f:
        f.write("name: my_strategy_test\nstrategy: my_strategy\n")

    is_valid, errors = validate_strategy_configs(str(src_dir), str(scenarios_dir))
    assert is_valid
    assert len(errors) == 0, f"Unexpected errors: {errors}"
