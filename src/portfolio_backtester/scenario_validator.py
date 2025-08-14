from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .interfaces import (
    StrategyResolverFactory,
    AllocationValidatorFactory,
    SignalValidatorFactory,
    UniversalConfigValidatorFactory,
)
from .yaml_validator import YamlError, YamlErrorType, YamlValidator

# Module-level logger
logger = logging.getLogger(__name__)

# Module-level polymorphic validators
_strategy_resolver = StrategyResolverFactory.create()
_allocation_validator = AllocationValidatorFactory.create()
_signal_validator = SignalValidatorFactory.create()
_universal_config_validator = UniversalConfigValidatorFactory.create()

"""Dynamic scenario YAML semantic validator.

This module validates backtest/optimizer scenario configuration files **after** they
have passed YAML-syntax validation.  It performs the following checks:

1. Common top-level keys that every scenario should (optionally) define
   are validated for correct *type* (e.g. integers for window sizes).
2. The referenced *strategy* is resolved using the existing ``_resolve_strategy``
   helper – an error is raised when the strategy cannot be found.
3. The scenario's ``optimizers`` section is inspected.  For every optimizer
   (``optuna`` or ``genetic`` today) we
      • verifies that the structure is a mapping,
      • checks that the compulsory ``optimize`` list is a list of mappings,
      • for every optimisation parameter checks that the referenced parameter
        is either tunable by the strategy **or** is present in the shared
        ``OPTIMIZER_PARAMETER_DEFAULTS`` dictionary (e.g. GA-specific knobs),
      • basic sanity checks on parameter specification fields (``type``,
        ``min_value``/``max_value`` etc.).
4. Keys given in ``strategy_params`` are verified against the strategy's
   ``tunable_parameters`` set.  A prefix of ``<strategy>.`` is tolerated and
   stripped during validation so that both
   ``momentum.lookback_months`` *and* ``lookback_months`` are accepted.

On validation failure a list of :class:`YamlError` instances is returned – the
calling code can format them via :pymeth:`YamlValidator.format_errors`.
"""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_ALLOWED_OPTIMIZERS: Set[str] = {"optuna", "genetic"}
_VALID_PARAM_TYPES: Set[str] = {"int", "float", "categorical", "str", "bool"}

_COMMON_INT_KEYS: Set[str] = {
    "train_window_months",
    "test_window_months",
}

_COMMON_STR_KEYS: Set[str] = {
    "rebalance_frequency",
    "position_sizer",
}


# ---------------------------------------------------------------------------
# Internal helper functions (extracted to reduce complexity)
# ---------------------------------------------------------------------------


def _ensure_mapping(
    scenario_data: Any, file_path_str: Optional[str]
) -> Tuple[Dict[str, Any], List["YamlError"]]:
    """Validate that scenario_data is a dict; return mapping and any errors."""
    errors: List[YamlError] = []
    if not _is_dict(scenario_data):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message=f"scenario_data is not a dictionary, got {type(scenario_data).__name__}: {repr(scenario_data)[:200]}",
                file_path=file_path_str,
            )
        )
        return {}, errors
    return scenario_data, errors


def _validate_common_keys(
    scenario_data: Dict[str, Any], file_path_str: Optional[str]
) -> List["YamlError"]:
    """Validate presence and basic types of common keys such as name and common ints/strs."""
    errors: List[YamlError] = []
    if "name" not in scenario_data or not _is_str(scenario_data.get("name")):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="Scenario is missing required key 'name' (string)",
                file_path=file_path_str,
            )
        )
    return errors


def _resolve_strategy_and_tunables(
    scenario_data: Dict[str, Any], file_path_str: Optional[str]
) -> Tuple[Dict[str, Any], List["YamlError"], Optional[type]]:
    """Resolve strategy class and obtain its tunable parameters; validate meta-strategy universe misuse."""
    errors: List[YamlError] = []
    strategy_tunable_params: Dict[str, Any] = {}

    if "strategy" not in scenario_data:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="Scenario is missing required key 'strategy'",
                file_path=file_path_str,
            )
        )
        return strategy_tunable_params, errors, None

    strat_cls = _strategy_resolver.resolve_strategy(scenario_data["strategy"])
    if strat_cls is None:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message=(
                    f"Unknown strategy '{scenario_data['strategy']}'. "
                    "Ensure the name matches a discovered class in built-ins or user strategies. "
                    "Discovered roots: strategies/builtins/{portfolio,signal,meta} and strategies/user/{portfolio,signal,meta}."
                ),
                file_path=file_path_str,
            )
        )
        return strategy_tunable_params, errors, None

    try:
        strategy_tunable_params = _strategy_resolver.tunable_parameters(strat_cls)
    except Exception as exc:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message=f"Failed to retrieve tunable parameters from strategy: {exc}",
                file_path=file_path_str,
            )
        )
        strategy_tunable_params = {}

    # Fallback: if no tunables were resolved, try direct registry lookup by name
    if not strategy_tunable_params and isinstance(scenario_data.get("strategy"), str):
        try:
            from .strategies._core.registry import get_strategy_registry

            reg = get_strategy_registry()
            cls = reg.get_strategy_class(scenario_data["strategy"])
            if cls is not None:
                fn = getattr(cls, "tunable_parameters", None)
                if callable(fn):
                    # Accept dict mapping or list/set of names
                    result = fn()
                    if isinstance(result, dict):
                        strategy_tunable_params = result
                    else:
                        strategy_tunable_params = {name: {} for name in list(result)}
        except Exception:
            # Ignore fallback errors; keep previous (possibly empty) mapping
            pass

    # Meta strategy universe constraints
    if _strategy_resolver.is_meta_strategy(strat_cls):
        if "universe_config" in scenario_data:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        "Meta strategies cannot define 'universe_config'. "
                        "Meta strategies inherit their universe from their sub-strategies."
                    ),
                    file_path=file_path_str,
                )
            )
        if "universe" in scenario_data:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        "Meta strategies cannot define 'universe'. "
                        "Meta strategies inherit their universe from their sub-strategies."
                    ),
                    file_path=file_path_str,
                )
            )

    return strategy_tunable_params, errors, strat_cls


def _validate_common_types_and_windows(
    scenario_data: Dict[str, Any], file_path_str: Optional[str]
) -> List["YamlError"]:
    """Validate common integer/string keys and logical window constraints."""
    errors: List[YamlError] = []
    for key in _COMMON_INT_KEYS.intersection(scenario_data.keys()):
        val = scenario_data[key]
        if not _is_int(val):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"'{key}' should be an integer (got {type(val).__name__})",
                    file_path=file_path_str,
                )
            )
            continue
        if val <= 0:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"'{key}' must be positive (got {val})",
                    file_path=file_path_str,
                )
            )
    if "train_window_months" in scenario_data and "test_window_months" in scenario_data:
        train = scenario_data["train_window_months"]
        test = scenario_data["test_window_months"]
        if _is_int(train) and _is_int(test) and (train + test) <= 0:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message="Train and test windows sum to non-positive value",
                    file_path=file_path_str,
                )
            )
        # Add general advisories for short windows (used in tests)
        if _is_int(train) and train < 6:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"train_window_months ({train}) is too short for meaningful diagnostics",
                    file_path=file_path_str,
                )
            )
        if _is_int(test) and test < 3:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"test_window_months ({test}) may be too short to evaluate performance",
                    file_path=file_path_str,
                )
            )
    for key in _COMMON_STR_KEYS.intersection(scenario_data.keys()):
        if not _is_str(scenario_data[key]):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"'{key}' should be a string (got {type(scenario_data[key]).__name__})",
                    file_path=file_path_str,
                )
            )
    return errors


def _validate_optimizers_section(
    scenario_data: Dict[str, Any],
    strategy_tunable_params: Dict[str, Any],
    optimizer_parameter_defaults: Dict[str, Any],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate optimizers mapping and each optimize item."""
    errors: List[YamlError] = []
    optimizers_cfg = scenario_data.get("optimizers")
    if optimizers_cfg is None:
        return errors
    if not _is_dict(optimizers_cfg):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="'optimizers' section must be a mapping/dictionary",
                file_path=file_path_str,
            )
        )
        return errors

    for opt_name, opt_cfg in optimizers_cfg.items():
        if opt_name not in _ALLOWED_OPTIMIZERS:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Unsupported optimizer '{opt_name}'. Allowed: {_ALLOWED_OPTIMIZERS}",
                    file_path=file_path_str,
                )
            )
            continue
        if not _is_dict(opt_cfg):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Configuration of optimizer '{opt_name}' must be a mapping/dictionary",
                    file_path=file_path_str,
                )
            )
            continue

        optimize_list = opt_cfg.get("optimize", [])
        if not _is_list(optimize_list):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Optimizer '{opt_name}': 'optimize' must be a list of parameter specs",
                    file_path=file_path_str,
                )
            )
            continue

        for idx, param_spec in enumerate(optimize_list):
            if not _is_dict(param_spec):
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Optimizer '{opt_name}' – entry #{idx + 1}: each item must be a mapping"
                        ),
                        file_path=file_path_str,
                    )
                )
                continue

            param_name = param_spec.get("parameter")
            if not param_name:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Optimizer '{opt_name}' – entry #{idx + 1}: missing required key 'parameter'"
                        ),
                        file_path=file_path_str,
                    )
                )
                continue

            if (param_name not in strategy_tunable_params) and (
                param_name not in optimizer_parameter_defaults
            ):
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Parameter '{param_name}' (optimizer '{opt_name}') is not tunable by "
                            f"strategy and not present in OPTIMIZER_PARAMETER_DEFAULTS."
                        ),
                        file_path=file_path_str,
                    )
                )

            p_type = param_spec.get("type")
            if p_type is not None and p_type not in _VALID_PARAM_TYPES:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Parameter '{param_name}' – unsupported type '{p_type}'. "
                            f"Allowed: {_VALID_PARAM_TYPES}."
                        ),
                        file_path=file_path_str,
                    )
                )

            if (
                p_type in {"int", "float"}
                and "min_value" in param_spec
                and "max_value" in param_spec
            ):
                min_v = param_spec["min_value"]
                max_v = param_spec["max_value"]
                if not (_is_numeric(min_v) and _is_numeric(max_v)):
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=(
                                f"Parameter '{param_name}' – 'min_value'/'max_value' must be numbers"
                            ),
                            file_path=file_path_str,
                        )
                    )
                elif max_v < min_v:
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=(
                                f"Parameter '{param_name}' – 'max_value' ({max_v}) is smaller than 'min_value' ({min_v})"
                            ),
                            file_path=file_path_str,
                        )
                    )

                # Step size sanity check
                step_v = param_spec.get("step")
                if _is_numeric(step_v) and _is_numeric(min_v) and _is_numeric(max_v):
                    if step_v > (max_v - min_v):
                        errors.append(
                            YamlError(
                                error_type=YamlErrorType.VALIDATION_ERROR,
                                message=(
                                    f"Parameter '{param_name}' step size {step_v} is larger than range {max_v - min_v}"
                                ),
                                file_path=file_path_str,
                            )
                        )

    return errors


def _validate_strategy_params(
    scenario_data: Dict[str, Any],
    strategy_tunable_params: Dict[str, Any],
    file_path_str: Optional[str],
    strategy_class: Optional[type] = None,
) -> List["YamlError"]:
    """Validate that provided strategy_params correspond to tunables and pass type/range checks."""
    errors: List[YamlError] = []
    strategy_params_cfg = scenario_data.get("strategy_params", {})
    if strategy_params_cfg is not None and not _is_dict(strategy_params_cfg):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="'strategy_params' must be a mapping/dictionary",
                file_path=file_path_str,
            )
        )
        return errors

    if not _is_dict(strategy_params_cfg):
        return errors

    # Check if this is a meta strategy using the strategy resolver
    is_meta_strategy = _strategy_resolver.is_meta_strategy(strategy_class)

    provided_params = set()
    for raw_key, value in strategy_params_cfg.items():
        param_name = raw_key.split(".", 1)[-1]
        provided_params.add(param_name)

        # Skip allocations parameter for meta strategies - it's validated separately
        if is_meta_strategy and param_name == "allocations":
            continue

        if param_name not in strategy_tunable_params:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        f"strategy_params contains unknown parameter '{raw_key}'. "
                        f"Allowed parameters: {sorted(strategy_tunable_params.keys())}"
                    ),
                    file_path=file_path_str,
                )
            )
            continue

        meta = strategy_tunable_params[param_name]
        p_type = meta.get("type")
        if p_type and _is_str(p_type):
            try:
                expected_type = getattr(builtins, p_type, None)
                if expected_type and not isinstance(value, expected_type):
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=(
                                f"Parameter '{raw_key}' type mismatch: expected {p_type}, got {type(value).__name__}"
                            ),
                            file_path=file_path_str,
                        )
                    )
                    # Skip range checks when the type itself is wrong
                    continue
            except (AttributeError, TypeError):
                # Skip type checking if we can't resolve the type
                pass

        min_v = meta.get("min")
        if min_v is not None and value < min_v:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Parameter '{raw_key}' below optimization minimum: {value} < {min_v}",
                    file_path=file_path_str,
                )
            )
        max_v = meta.get("max")
        if max_v is not None and value > max_v:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Parameter '{raw_key}' above optimization maximum: {value} > {max_v}",
                    file_path=file_path_str,
                )
            )

        # Validate categorical choices if present in metadata
        choices = meta.get("choices") or meta.get("options")
        if choices is not None and isinstance(choices, (list, tuple)):
            if value not in choices:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Parameter '{raw_key}' value '{value}' not in optimization choices {list(choices)}"
                        ),
                        file_path=file_path_str,
                    )
                )

    # Detect missing parameters that are referenced by optimizers
    # and also enforce required tunables
    optimizer_params: Set[str] = set()
    for opt_cfg in (scenario_data.get("optimizers") or {}).values():
        if _is_dict(opt_cfg):
            for entry in opt_cfg.get("optimize", []):
                if _is_dict(entry) and entry.get("parameter"):
                    optimizer_params.add(str(entry["parameter"]))

    for param, meta in strategy_tunable_params.items():
        if meta.get("required", False) and param not in provided_params:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Required parameter '{param}' missing in strategy_params",
                    file_path=file_path_str,
                )
            )
    # Any optimizer param missing from strategy_params should produce a helpful message
    for opt_param in optimizer_params:
        # allow non-strategy global parameters from OPTIMIZER_PARAMETER_DEFAULTS; handled earlier
        if opt_param in strategy_tunable_params and opt_param not in provided_params:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"Optimization parameter '{opt_param}' is missing from strategy_params",
                    file_path=file_path_str,
                )
            )

    return errors


def _validate_universe_config(
    scenario_data: Dict[str, Any],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate universe_config structure when present."""
    errors: List[YamlError] = []
    universe_config = scenario_data.get("universe_config")
    if universe_config is None:
        return errors

    allowed_types = {"single_symbol", "fixed", "named", "method"}
    universe_type = universe_config.get("type")
    if universe_type not in allowed_types:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message=f"universe_config.type '{universe_type}' is not supported. Allowed types: {sorted(allowed_types)}",
                file_path=file_path_str,
            )
        )
    if universe_type == "single_symbol" and "ticker" not in universe_config:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="single_symbol universe_config requires a 'ticker' field",
                file_path=file_path_str,
            )
        )
    return errors


def _validate_dynamic_scenario_logic(
    scenario_data: Dict[str, Any],
    strategy_tunable_params: Dict[str, Any],
    strategy_class: Optional[type],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Dynamic validation for all scenario types with deep logical consistency checks."""
    errors: List[YamlError] = []

    # Dynamic strategy type detection
    strategy_type = _detect_strategy_type(strategy_class)

    # Apply strategy-type-specific validation
    errors.extend(
        _validate_strategy_type_specific_logic(
            scenario_data, strategy_type, strategy_class, file_path_str
        )
    )

    # Apply universal validation rules
    errors.extend(
        _validate_universal_configuration_logic(
            scenario_data, strategy_tunable_params, strategy_class, file_path_str
        )
    )

    return errors


def _detect_strategy_type(strategy_class: Optional[type]) -> str:
    """Dynamically detect the type of strategy based on class hierarchy using polymorphic resolver."""
    return _strategy_resolver.detect_strategy_type(strategy_class)


def _validate_strategy_type_specific_logic(
    scenario_data: Dict[str, Any],
    strategy_type: str,
    strategy_class: Optional[type],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Apply validation rules specific to the detected strategy type."""
    errors: List[YamlError] = []

    if strategy_type == "meta":
        errors.extend(_validate_meta_strategy_logic(scenario_data, strategy_class, file_path_str))
    elif strategy_type == "portfolio":
        errors.extend(
            _validate_portfolio_strategy_logic(scenario_data, strategy_class, file_path_str)
        )
    elif strategy_type == "signal":
        errors.extend(_validate_signal_strategy_logic(scenario_data, strategy_class, file_path_str))
    # No dedicated diagnostic branch: diagnostic utilities are no longer treated as strategies

    return errors


def _validate_meta_strategy_logic(
    scenario_data: Dict[str, Any],
    strategy_class: Optional[type],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate meta strategy specific configuration using polymorphic validator."""
    return _allocation_validator.validate_meta_strategy_logic(
        scenario_data, strategy_class, file_path_str
    )


def _validate_meta_strategy_allocations(
    allocations: Any,
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate meta strategy allocations structure using polymorphic validator."""
    return _allocation_validator.validate_allocations(allocations, file_path_str)


def _validate_portfolio_strategy_logic(
    scenario_data: Dict[str, Any],
    strategy_class: Optional[type],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate portfolio strategy specific configuration."""
    errors: List[YamlError] = []

    # Portfolio strategies typically require universe configuration
    if "universe_config" not in scenario_data and "universe" not in scenario_data:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="Portfolio strategies typically require 'universe_config' or 'universe' specification",
                file_path=file_path_str,
            )
        )

    # Check for common portfolio strategy parameters
    strategy_params = scenario_data.get("strategy_params", {})
    strategy_name = scenario_data.get("strategy", "")

    # Look for momentum-related parameters
    momentum_params = ["lookback_months", "rolling_window", "num_holdings"]
    momentum_param_count = sum(
        1 for param in momentum_params if f"{strategy_name}.{param}" in strategy_params
    )

    if momentum_param_count > 0 and momentum_param_count < 2:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message=f"Momentum-based portfolio strategy appears incomplete. Found {momentum_param_count} of common momentum parameters: {momentum_params}",
                file_path=file_path_str,
            )
        )

    return errors


def _validate_signal_strategy_logic(
    scenario_data: Dict[str, Any],
    strategy_class: Optional[type],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate signal strategy specific configuration using polymorphic validator."""
    return _signal_validator.validate_signal_strategy_logic(
        scenario_data, strategy_class, file_path_str
    )


def _validate_diagnostic_strategy_logic(*args: Any, **kwargs: Any) -> List["YamlError"]:
    """Deprecated: diagnostics are not strategies; retained as no-op for compatibility."""
    return []


def _validate_timing_config(
    timing_config: Dict[str, Any],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Validate timing configuration structure using polymorphic validator."""
    return _signal_validator.validate_timing_config(timing_config, file_path_str)


def _validate_universal_configuration_logic(
    scenario_data: Dict[str, Any],
    strategy_tunable_params: Dict[str, Any],
    strategy_class: Optional[type],
    file_path_str: Optional[str],
) -> List["YamlError"]:
    """Apply universal validation rules that apply to all strategy types using polymorphic validator."""
    return _universal_config_validator.validate_universal_configuration_logic(
        scenario_data, strategy_tunable_params, strategy_class, file_path_str
    )


# ---------------------------------------------------------------------------
# Type checking helper functions (to replace isinstance violations)
# ---------------------------------------------------------------------------


def _is_dict(obj: Any) -> bool:
    """Check if object is a dict."""
    return isinstance(obj, dict)


def _is_list(obj: Any) -> bool:
    """Check if object is a list."""
    return isinstance(obj, list)


def _is_str(obj: Any) -> bool:
    """Check if object is a string."""
    return isinstance(obj, str)


def _is_int(obj: Any) -> bool:
    """Check if object is an integer."""
    return isinstance(obj, int)


def _is_numeric(obj: Any) -> bool:
    """Check if object is numeric."""
    return isinstance(obj, (int, float))


def _is_set(obj: Any) -> bool:
    """Check if object is a set."""
    return isinstance(obj, set)


# ---------------------------------------------------------------------------
# Public validation helpers
# ---------------------------------------------------------------------------


def validate_scenario_semantics(
    scenario_data: Dict[str, Any],
    optimizer_parameter_defaults: Dict[str, Any] | None = None,
    *,
    file_path: str | Path | None = None,
) -> List[YamlError]:
    """Validate *semantic* correctness of a scenario.

    Parameters
    ----------
    scenario_data
        Parsed YAML mapping representing **one** scenario file.
    optimizer_parameter_defaults
        The ``OPTIMIZER_PARAMETER_DEFAULTS`` mapping loaded from
        ``parameters.yaml``.  This is used to recognise GA-specific or other
        global optimisation parameters that are not part of the strategy's
        tunable parameters.
    file_path
        Path of the file currently validated – only used for error messages.

    Returns
    -------
    list[YamlError]
        Empty when *no* semantic issues were found.
    """

    logger.debug("validate_scenario_semantics called with %s", scenario_data)
    errors: List[YamlError] = []
    file_path_str = str(file_path) if file_path is not None else None

    # 0) Ensure mapping
    scenario_mapping, map_errors = _ensure_mapping(scenario_data, file_path_str)
    if map_errors:
        return map_errors
    scenario_data = scenario_mapping

    # 1) Mandatory keys & basic 'name'
    errors.extend(_validate_common_keys(scenario_data, file_path_str))

    # -------------------------------------------------------------------
    # 1. Mandatory keys & basic types
    # -------------------------------------------------------------------
    if "name" not in scenario_data or not _is_str(scenario_data["name"]):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="Scenario is missing required key 'name' (string)",
                file_path=file_path_str,
            )
        )

    # 2) Resolve strategy and tunables (+ meta constraints)
    strategy_tunable_params, strat_errors, strat_cls = _resolve_strategy_and_tunables(
        scenario_data, file_path_str
    )
    errors.extend(strat_errors)

    # Provide a safe empty mapping if not supplied
    if optimizer_parameter_defaults is None:
        optimizer_parameter_defaults = {}

    # 3) Common type checks and window logic
    errors.extend(_validate_common_types_and_windows(scenario_data, file_path_str))

    # 4) Optimizers section validation
    errors.extend(
        _validate_optimizers_section(
            scenario_data,
            strategy_tunable_params,
            optimizer_parameter_defaults,
            file_path_str,
        )
    )

    # 5) Strategy params validation (uniform; no special diagnostic leniency)
    errors.extend(
        _validate_strategy_params(scenario_data, strategy_tunable_params, file_path_str, strat_cls)
    )

    # -------------------------------------------------------------------
    # 4. Validate strategy_params – keys must correspond to tunable params
    # -------------------------------------------------------------------
    strategy_params_cfg = scenario_data.get("strategy_params", {})
    if strategy_params_cfg is not None and not _is_dict(strategy_params_cfg):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="'strategy_params' must be a mapping/dictionary",
                file_path=file_path_str,
            )
        )
    elif _is_dict(strategy_params_cfg):
        logger.debug("strategy_params_cfg: %s", strategy_params_cfg)
        provided_params = set()
        logger.debug("Entering strategy_params_cfg loop: %s", strategy_params_cfg)
        # Check if this is a meta strategy using the strategy resolver
        is_meta_strategy_duplicate = _strategy_resolver.is_meta_strategy(strat_cls)

        for raw_key, value in strategy_params_cfg.items():
            logger.debug("Loop body: raw_key=%s, value=%s", raw_key, value)
            param_name = raw_key.split(".", 1)[-1]
            provided_params.add(param_name)
            logger.debug(
                "param_name=%s, strategy_tunable_params=%s",
                param_name,
                list(strategy_tunable_params.keys()),
            )

            # Skip allocations parameter for meta strategies - it's validated separately
            if is_meta_strategy_duplicate and param_name == "allocations":
                continue

            if param_name not in strategy_tunable_params:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"strategy_params contains unknown parameter '{raw_key}'. "
                            f"Allowed parameters: {sorted(strategy_tunable_params.keys())}"
                        ),
                        file_path=file_path_str,
                    )
                )
                continue
            meta = strategy_tunable_params[param_name]
            p_type = meta.get("type")
            logger.debug(
                "param_name=%s, p_type=%s, value=%s, value_type=%s",
                param_name,
                p_type,
                value,
                type(value).__name__,
            )
            if p_type and _is_str(p_type):
                try:
                    expected_type = getattr(builtins, p_type, None)
                    logger.debug("expected_type=%s", expected_type)
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(
                            YamlError(
                                error_type=YamlErrorType.VALIDATION_ERROR,
                                message=(
                                    f"Parameter '{raw_key}' type mismatch: expected {p_type}, got {type(value).__name__}"
                                ),
                                file_path=file_path_str,
                            )
                        )
                        # Skip range checks when the type itself is wrong
                        continue
                except (AttributeError, TypeError):
                    # Skip type checking if we can't resolve the type
                    pass

            min_v = meta.get("min")
            logger.debug("Parameter %s, value=%s, min=%s, meta=%s", raw_key, value, min_v, meta)
            logger.debug("Range check for %s: min_v=%s, value=%s", raw_key, min_v, value)
            if min_v is not None and value < min_v:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=f"Parameter '{raw_key}' below min: {value} < {min_v}",
                        file_path=file_path_str,
                    )
                )
            max_v = meta.get("max")
            if max_v is not None and value > max_v:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=f"Parameter '{raw_key}' above max: {value} > {max_v}",
                        file_path=file_path_str,
                    )
                )
        # Check for missing required params
        for param, meta in strategy_tunable_params.items():
            if meta.get("required", False) and param not in provided_params:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=f"Required parameter '{param}' missing in strategy_params",
                        file_path=file_path_str,
                    )
                )

    # 6) Universe config validation
    errors.extend(_validate_universe_config(scenario_data, file_path_str))

    # 7) Enhanced dynamic validation for all scenario types
    errors.extend(
        _validate_dynamic_scenario_logic(
            scenario_data, strategy_tunable_params, strat_cls, file_path_str
        )
    )

    return errors


# ---------------------------------------------------------------------------
# Convenience helper for external callers
# ---------------------------------------------------------------------------


def validate_scenario_file(
    file_path: Path | str,
    optimizer_parameter_defaults: Dict[str, Any] | None = None,
) -> List[YamlError]:
    """Validate a scenario *file* including YAML-syntax and semantics."""
    logger.debug("validate_scenario_file called with %s", file_path)
    file_path = Path(file_path)

    # First – YAML syntax via existing validator
    syntax_validator = YamlValidator()
    is_valid_yaml, data, syntax_errors = syntax_validator.validate_file(file_path)

    # If syntax failed we can stop early – just return syntax errors
    if not is_valid_yaml or data is None:
        return syntax_errors

    # Then – semantic validation
    semantic_errors = validate_scenario_semantics(
        data,
        optimizer_parameter_defaults=optimizer_parameter_defaults,
        file_path=file_path,
    )

    return syntax_errors + semantic_errors
