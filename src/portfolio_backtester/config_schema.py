"""Schema validation for top-level strategy scenario YAML files.

This lightweight validator focuses on the `strategy` and `strategy_params` blocks
and enforces the new *namespaced* parameter convention introduced during the
parameter-prefix refactor (e.g. `momentum.lookback_months`).

The file purposefully limits scope: timing-related settings are validated by
`portfolio_backtester.timing.config_schema.TimingConfigSchema`, while optimizer
specifications are validated inside the optimization subsystem.  Here we only
check structural correctness of scenario files and the prefix rule.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    field: str
    value: Any
    message: str
    suggestion: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"


class StrategyConfigSchema:
    """Validator enforcing namespaced `strategy_params` keys."""

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        errors: List[ValidationError] = []

        # Basic presence checks -------------------------------------------------
        strategy = config.get("strategy")
        if not strategy or not isinstance(strategy, str):
            errors.append(
                ValidationError(
                    field="strategy",
                    value=strategy,
                    message="Missing or invalid 'strategy' field (must be string)",
                    suggestion="Add e.g. strategy: momentum"
                )
            )
            # Can't continue prefix checks without strategy name
            return errors

        params = config.get("strategy_params")
        if params is None:
            errors.append(
                ValidationError(
                    field="strategy_params",
                    value=None,
                    message="Missing required 'strategy_params' section",
                    suggestion="Add strategy_params: {...}"
                )
            )
            return errors
        if not isinstance(params, dict):
            errors.append(
                ValidationError(
                    field="strategy_params",
                    value=type(params).__name__,
                    message="'strategy_params' must be a mapping/dictionary",
                )
            )
            return errors

        # Prefix validation -----------------------------------------------------
        prefix = f"{strategy}."
        for key in params.keys():
            if key.startswith(prefix):
                continue  # Correctly prefixed
            if "." in key:
                # Namespaced for some other purpose – allow but warn
                errors.append(
                    ValidationError(
                        field=f"strategy_params.{key}",
                        value=None,
                        message=(
                            "Parameter key contains a dot but does not start with the "
                            f"strategy prefix '{prefix}'. This may be unintended."
                        ),
                        severity="warning",
                    )
                )
            else:
                # Plain key – invalid after refactor
                errors.append(
                    ValidationError(
                        field=f"strategy_params.{key}",
                        value=None,
                        message="Parameter key missing required '<strategy>.' prefix",
                        suggestion=f"Rename to '{prefix}{key}'"
                    )
                )

        return errors

    # ---------------------------------------------------------------------
    # Convenience wrappers
    # ---------------------------------------------------------------------
    @classmethod
    def validate_yaml_file(cls, file_path: Union[str, Path]) -> List[ValidationError]:
        path = Path(file_path)
        if not path.exists():
            return [ValidationError(field="file", value=str(path), message="File not found")]
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            return [ValidationError(field="yaml", value=str(exc), message="YAML parsing error")]
        except Exception as exc:
            return [ValidationError(field="file", value=str(exc), message="File read error")]

        if not isinstance(obj, dict):
            return [ValidationError(field="yaml", value="root", message="Root of YAML must be a mapping")]

        return cls.validate_config(obj)

    @classmethod
    def format_report(cls, errors: List[ValidationError]) -> str:
        if not errors:
            return "✓ Configuration is valid"
        parts = ["Configuration Validation Report", "=" * 35, ""]
        for idx, err in enumerate(errors, 1):
            sym = "✗" if err.severity == "error" else "⚠"
            parts.append(f"{idx}. {sym} {err.field}: {err.message}")
            if err.suggestion:
                parts.append(f"   Suggestion: {err.suggestion}")
            parts.append("")
        return "\n".join(parts)