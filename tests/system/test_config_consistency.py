import os
from pathlib import Path
import re
import unittest
import pytest
import yaml


@pytest.mark.system
class TestConfigConsistency(unittest.TestCase):

    def test_strategy_has_scenario(self):
        """Check if each strategy has at least one corresponding scenario file."""
        strategies_dir = (
            Path(__file__).parent.parent.parent / "src" / "portfolio_backtester" / "strategies"
        )
        scenarios_dir = Path(__file__).parent.parent.parent / "config" / "scenarios"

        # Exclude utility classes, abstract classes, base classes, and internal refactor dirs
        excluded_files = {
            "strategy_factory",
            "leverage_and_smoothing",
            "candidate_weights",
            "signal_generator",
            "ema_signal_generator",
            "rsi_calculator",
            "price_data_processor",
            "base_portfolio_momentum_strategy",
            "portfolio_momentum_strategy",
            "portfolio_momentum",
            # momentum framework base and internal variants (no direct scenarios)
            "base_momentum_portfolio_strategy",
            "fixed_weight_portfolio_strategy",
            "momentum_unfiltered_atr_portfolio_strategy",
            "momentum_beta_filtered_portfolio_strategy",
            "momentum_dvol_sizer_portfolio_strategy",
            # internal registry modules (new layout)
            "strategy_registry",
            "solid_strategy_registry",
            "strategy_factory_impl",
            "strategy_validator",
        }

        def is_internal_path(path: Path) -> bool:
            # Ignore the new internal dirs that do not correspond to user-invocable strategies
            parts = [p.lower() for p in path.parts]
            return any(seg in parts for seg in {"_core", "builtins", "registry", "examples"})

        def camel_to_snake(name: str) -> str:
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
            return snake

        def normalize_any(name: str) -> str:
            # Convert CamelCase to snake_case and strip trailing _strategy
            if any(ch.isupper() for ch in name):
                name = camel_to_snake(name)
            name = name.lower()
            if name.endswith("_strategy"):
                name = name[: -len("_strategy")]
            # Normalize legacy ordering variants to canonical names
            replacements = {
                "beta_filtered_momentum_portfolio": "momentum_beta_filtered_portfolio",
                "dvol_sizer_momentum_portfolio": "momentum_dvol_sizer_portfolio",
                "unfiltered_atr_momentum_portfolio": "momentum_unfiltered_atr_portfolio",
            }
            name = replacements.get(name, name)
            return name

        strategy_files = {
            normalize_any(f.stem)
            for f in strategies_dir.glob("**/*.py")
            if f.stem != "__init__"
            and not f.parent.name == "base"
            and not is_internal_path(f)
            and f.stem not in excluded_files
        }
        scenario_strategies = set()

        # Walk through subdirectories to find scenario files
        for root, dirs, files in os.walk(scenarios_dir):
            for scenario_file in files:
                if scenario_file.endswith(".yaml"):
                    scenario_path = Path(root) / scenario_file
                    with open(scenario_path, "r") as f:
                        scenario_data = yaml.safe_load(f)
                        if (
                            scenario_data
                            and isinstance(scenario_data, dict)
                            and "strategy" in scenario_data
                        ):
                            strategy_val = scenario_data["strategy"]
                            if isinstance(strategy_val, dict):
                                strategy_name = strategy_val.get("type")
                            else:
                                strategy_name = strategy_val
                            if strategy_name:
                                scenario_strategies.add(normalize_any(strategy_name))

        missing_scenarios = strategy_files - scenario_strategies
        self.assertEqual(
            len(missing_scenarios), 0, f"Strategies with missing scenarios: {missing_scenarios}"
        )


if __name__ == "__main__":
    unittest.main()
