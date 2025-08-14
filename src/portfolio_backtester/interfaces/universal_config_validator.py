"""
Universal config validator interface for configuration validation.

This module provides interfaces for validating universal configuration rules
that apply to all strategy types without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from portfolio_backtester.yaml_validator import YamlError, YamlErrorType


class IUniversalConfigValidator(ABC):
    """Interface for validating universal configuration rules."""

    @abstractmethod
    def validate_universal_configuration_logic(
        self,
        scenario_data: Dict[str, Any],
        strategy_tunable_params: Dict[str, Any],
        strategy_class: Any,
        file_path_str: Optional[str],
    ) -> List[Any]:
        """Apply universal validation rules that apply to all strategy types."""
        pass


class DefaultUniversalConfigValidator(IUniversalConfigValidator):
    """Default implementation of universal config validator."""

    def validate_universal_configuration_logic(
        self,
        scenario_data: Dict[str, Any],
        strategy_tunable_params: Dict[str, Any],
        strategy_class: Any,
        file_path_str: Optional[str],
    ) -> List[Any]:
        """Apply universal validation rules that apply to all strategy types."""
        # YamlError and YamlErrorType imported at module level
        errors = []

        # Validate rebalance_frequency
        rebalance_frequency = scenario_data.get("rebalance_frequency")
        if rebalance_frequency is not None:
            valid_frequencies = {"D", "ME", "MS", "QE", "QS", "YE", "YS", "W"}
            if rebalance_frequency not in valid_frequencies:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Invalid rebalance_frequency '{rebalance_frequency}'. "
                            f"Valid frequencies are: {', '.join(valid_frequencies)}."
                        ),
                        file_path=file_path_str,
                    )
                )

        # Validate position_sizer
        position_sizer = scenario_data.get("position_sizer")
        if position_sizer is not None:
            valid_sizers = {
                "equal_weight",
                "volatility_weighted",
                "market_cap_weighted",
                "custom",
            }
            if position_sizer not in valid_sizers:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Invalid position_sizer '{position_sizer}'. "
                            f"Valid sizers are: {', '.join(valid_sizers)}."
                        ),
                        file_path=file_path_str,
                    )
                )

        # Validate window sizes
        train_window = scenario_data.get("train_window_months")
        if isinstance(train_window, int) and train_window < 6:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        f"train_window_months ({train_window}) is too short. "
                        "Recommend at least 6 months for meaningful results."
                    ),
                    file_path=file_path_str,
                )
            )

        # Validate optimization targets
        optimization_targets = scenario_data.get("optimization_targets", [])
        if isinstance(optimization_targets, list):
            for target in optimization_targets:
                if isinstance(target, dict):
                    metric = target.get("name")
                    direction = target.get("direction")

                    # Check metric direction recommendations
                    if metric and direction:
                        # Metrics that should typically be maximized
                        maximize_metrics = {
                            "Sharpe",
                            "Sortino",
                            "Calmar",
                            "CAGR",
                            "AnnualizedReturn",
                        }
                        # Metrics that should typically be minimized
                        minimize_metrics = {
                            "MaxDrawdown",
                            "Volatility",
                            "Downside",
                            "UlcerIndex",
                        }

                        if metric in maximize_metrics and direction == "minimize":
                            errors.append(
                                YamlError(
                                    error_type=YamlErrorType.VALIDATION_ERROR,
                                    message=(
                                        f"Metric '{metric}' should typically be maximized, not minimized. "
                                        "Please check if this is intentional."
                                    ),
                                    file_path=file_path_str,
                                )
                            )
                        elif metric in minimize_metrics and direction == "maximize":
                            errors.append(
                                YamlError(
                                    error_type=YamlErrorType.VALIDATION_ERROR,
                                    message=(
                                        f"Metric '{metric}' should typically be minimized, not maximized. "
                                        "Please check if this is intentional."
                                    ),
                                    file_path=file_path_str,
                                )
                            )

        return errors


class UniversalConfigValidatorFactory:
    """Factory for creating universal config validators."""

    @staticmethod
    def create() -> IUniversalConfigValidator:
        """Create a new universal config validator instance."""
        return DefaultUniversalConfigValidator()
