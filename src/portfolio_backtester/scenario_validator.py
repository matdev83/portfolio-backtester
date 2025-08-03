from __future__ import annotations

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

from pathlib import Path
from typing import Any, Dict, List, Set
import builtins
import logging

logger = logging.getLogger(__name__)

from .yaml_validator import YamlError, YamlErrorType, YamlValidator
from .utils import _resolve_strategy

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
    
    # Debug: Check if scenario_data is actually a dictionary
    if not isinstance(scenario_data, dict):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message=f"scenario_data is not a dictionary, got {type(scenario_data).__name__}: {repr(scenario_data)[:200]}",
                file_path=file_path_str,
            )
        )
        return errors

    # -------------------------------------------------------------------
    # 1. Mandatory keys & basic types
    # -------------------------------------------------------------------
    if "name" not in scenario_data or not isinstance(scenario_data["name"], str):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="Scenario is missing required key 'name' (string)",
                file_path=file_path_str,
            )
        )

    if "strategy" not in scenario_data:
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="Scenario is missing required key 'strategy'",
                file_path=file_path_str,
            )
        )
        strategy_tunable_params: Dict[str, Any] = {}
    else:
        # Resolve strategy – ensures import works and gives access to tunable params
        strat_cls = _resolve_strategy(scenario_data["strategy"])
        if strat_cls is None:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        f"Unknown strategy '{scenario_data['strategy']}'. "
                        "Ensure the strategy name matches an existing implementation."
                    ),
                    file_path=file_path_str,
                )
            )
            strategy_tunable_params = {}
        else:
            # Strategy found – get declared tunable parameters
            try:
                strategy_tunable_params = strat_cls.tunable_parameters()
                if isinstance(strategy_tunable_params, set):
                    strategy_tunable_params = {p: {} for p in strategy_tunable_params}  # Backward compat
                elif not isinstance(strategy_tunable_params, dict):
                    raise ValueError('tunable_parameters must return dict')
            except Exception as exc:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=f'Failed to retrieve tunable parameters from strategy: {exc}',
                        file_path=file_path_str,
                    )
                )
                strategy_tunable_params = {}
            
            # Check if this is a meta strategy that incorrectly defines universe
            from .strategies.base.meta_strategy import BaseMetaStrategy
            if issubclass(strat_cls, BaseMetaStrategy):
                # Meta strategies should not have universe_config defined
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
                # Also check for legacy 'universe' key
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

    # Provide a safe empty mapping if not supplied
    if optimizer_parameter_defaults is None:
        optimizer_parameter_defaults = {}

    # -------------------------------------------------------------------
    # 2. Common keys – type checks only (fail-soft)
    # -------------------------------------------------------------------
    for key in _COMMON_INT_KEYS.intersection(scenario_data.keys()):
        val = scenario_data[key]
        if not isinstance(val, int):
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
    # Additional logical check for windows
    if 'train_window_months' in scenario_data and 'test_window_months' in scenario_data:
        train = scenario_data['train_window_months']
        test = scenario_data['test_window_months']
        if isinstance(train, int) and isinstance(test, int) and train + test <= 0:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message='Train and test windows sum to non-positive value',
                    file_path=file_path_str,
                )
            )

    for key in _COMMON_STR_KEYS.intersection(scenario_data.keys()):
        if not isinstance(scenario_data[key], str):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=f"'{key}' should be a string (got {type(scenario_data[key]).__name__})",
                    file_path=file_path_str,
                )
            )

    # -------------------------------------------------------------------
    # 3. Validate optimizers section (if present)
    # -------------------------------------------------------------------
    optimizers_cfg = scenario_data.get("optimizers")
    if optimizers_cfg is not None:
        if not isinstance(optimizers_cfg, dict):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message="'optimizers' section must be a mapping/dictionary",
                    file_path=file_path_str,
                )
            )
        else:
            for opt_name, opt_cfg in optimizers_cfg.items():
                if opt_name not in _ALLOWED_OPTIMIZERS:
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=f"Unsupported optimizer '{opt_name}'. Allowed: {_ALLOWED_OPTIMIZERS}",
                            file_path=file_path_str,
                        )
                    )
                    # Skip deeper checks for unknown optimizer
                    continue

                if not isinstance(opt_cfg, dict):
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=f"Configuration of optimizer '{opt_name}' must be a mapping/dictionary",
                            file_path=file_path_str,
                        )
                    )
                    continue

                # Optimizer-specific: verify 'optimize' list
                optimize_list = opt_cfg.get("optimize", [])
                if not isinstance(optimize_list, list):
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=f"Optimizer '{opt_name}': 'optimize' must be a list of parameter specs",
                            file_path=file_path_str,
                        )
                    )
                else:
                    for idx, param_spec in enumerate(optimize_list):
                        if not isinstance(param_spec, dict):
                            errors.append(
                                YamlError(
                                    error_type=YamlErrorType.VALIDATION_ERROR,
                                    message=(
                                        f"Optimizer '{opt_name}' – entry #{idx+1}: each item must be a mapping"
                                    ),
                                    file_path=file_path_str,
                                )
                            )
                            continue
                        # Mandatory 'parameter'
                        param_name = param_spec.get("parameter")
                        if not param_name:
                            errors.append(
                                YamlError(
                                    error_type=YamlErrorType.VALIDATION_ERROR,
                                    message=(
                                        f"Optimizer '{opt_name}' – entry #{idx+1}: missing required key 'parameter'"
                                    ),
                                    file_path=file_path_str,
                                )
                            )
                            continue

                        # Validate parameter existence
                        if (
                            param_name not in strategy_tunable_params
                            and param_name not in optimizer_parameter_defaults
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

                        # Basic checks on specification fields (if provided)
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

                        # Numeric range checks when min/max provided
                        if (
                            p_type in {"int", "float"}
                            and "min_value" in param_spec
                            and "max_value" in param_spec
                        ):
                            min_v = param_spec["min_value"]
                            max_v = param_spec["max_value"]
                            if not (isinstance(min_v, (int, float)) and isinstance(max_v, (int, float))):
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

    # -------------------------------------------------------------------
    # 4. Validate strategy_params – keys must correspond to tunable params
    # -------------------------------------------------------------------
    strategy_params_cfg = scenario_data.get("strategy_params", {})
    if strategy_params_cfg is not None and not isinstance(strategy_params_cfg, dict):
        errors.append(
            YamlError(
                error_type=YamlErrorType.VALIDATION_ERROR,
                message="'strategy_params' must be a mapping/dictionary",
                file_path=file_path_str,
            )
        )
    elif isinstance(strategy_params_cfg, dict):
        logger.debug("strategy_params_cfg: %s", strategy_params_cfg)
        provided_params = set()
        logger.debug("Entering strategy_params_cfg loop: %s", strategy_params_cfg)
        for raw_key, value in strategy_params_cfg.items():
            logger.debug("Loop body: raw_key=%s, value=%s", raw_key, value)
            param_name = raw_key.split('.', 1)[-1]
            provided_params.add(param_name)
            logger.debug("param_name=%s, strategy_tunable_params=%s", param_name, list(strategy_tunable_params.keys()))
            if param_name not in strategy_tunable_params:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(f'strategy_params contains unknown parameter \'{raw_key}\'. '
                                 f'Allowed parameters: {sorted(strategy_tunable_params.keys())}'),
                        file_path=file_path_str,
                    )
                )
                continue
            meta = strategy_tunable_params[param_name]
            p_type = meta.get('type')
            logger.debug("param_name=%s, p_type=%s, value=%s, value_type=%s", param_name, p_type, value, type(value).__name__)
            if p_type and isinstance(p_type, str):
                try:
                    expected_type = getattr(builtins, p_type, None)
                    logger.debug("expected_type=%s", expected_type)
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(
                            YamlError(
                                error_type=YamlErrorType.VALIDATION_ERROR,
                                message=(f"Parameter '{raw_key}' type mismatch: expected {p_type}, got {type(value).__name__}"),
                                file_path=file_path_str,
                            )
                        )
                        # Skip range checks when the type itself is wrong
                        continue
                except (AttributeError, TypeError):
                    # Skip type checking if we can't resolve the type
                    pass

            min_v = meta.get('min')
            logger.debug("Parameter %s, value=%s, min=%s, meta=%s", raw_key, value, min_v, meta)
            logger.debug("Range check for %s: min_v=%s, value=%s", raw_key, min_v, value)
            if min_v is not None and value < min_v:
                errors.append(YamlError(error_type=YamlErrorType.VALIDATION_ERROR, message=f'Parameter \'{raw_key}\' below min: {value} < {min_v}', file_path=file_path_str))
            max_v = meta.get('max')
            if max_v is not None and value > max_v:
                errors.append(YamlError(error_type=YamlErrorType.VALIDATION_ERROR, message=f'Parameter \'{raw_key}\' above max: {value} > {max_v}', file_path=file_path_str))
        # Check for missing required params
        for param, meta in strategy_tunable_params.items():
            if meta.get('required', False) and param not in provided_params:
                errors.append(YamlError(error_type=YamlErrorType.VALIDATION_ERROR, message=f'Required parameter \'{param}\' missing in strategy_params', file_path=file_path_str))

    # Validate universe_config type if present
    universe_config = scenario_data.get("universe_config")
    if universe_config is not None:
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
