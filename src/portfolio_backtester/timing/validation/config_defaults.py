"""
Default configuration generation for timing framework.
Provides default configurations for different timing modes.
"""

from typing import Dict, Any


class ConfigDefaults:
    """Generates default configurations for timing modes."""

    # Base configuration common to all modes
    BASE_CONFIG = {"enable_logging": False, "log_level": "INFO"}

    @classmethod
    def get_default_config(cls, mode: str = "time_based") -> Dict[str, Any]:
        """
        Get default configuration for a timing mode.

        Args:
            mode: Timing mode ('time_based', 'signal_based', 'custom')

        Returns:
            Default configuration dictionary

        Raises:
            ValueError: If mode is not recognized
        """
        if mode == "time_based":
            return cls._get_time_based_defaults()
        elif mode == "signal_based":
            return cls._get_signal_based_defaults()
        elif mode == "custom":
            return cls._get_custom_defaults()
        else:
            raise ValueError(f"Unknown timing mode: {mode}")

    @classmethod
    def _get_time_based_defaults(cls) -> Dict[str, Any]:
        """Get default time-based configuration."""
        return {
            "mode": "time_based",
            "rebalance_frequency": "M",
            "rebalance_offset": 0,
            **cls.BASE_CONFIG,
        }

    @classmethod
    def _get_signal_based_defaults(cls) -> Dict[str, Any]:
        """Get default signal-based configuration."""
        return {
            "mode": "signal_based",
            "scan_frequency": "D",
            "min_holding_period": 1,
            "max_holding_period": None,
            **cls.BASE_CONFIG,
        }

    @classmethod
    def _get_custom_defaults(cls) -> Dict[str, Any]:
        """Get default custom configuration."""
        return {
            "mode": "custom",
            "custom_controller_class": "your.module.CustomTimingController",
            "custom_controller_params": {},
            **cls.BASE_CONFIG,
        }

    @classmethod
    def get_all_supported_modes(cls) -> list[str]:
        """Get list of all supported timing modes."""
        return ["time_based", "signal_based", "custom"]

    @classmethod
    def get_example_config(cls, mode: str) -> str:
        """
        Get example YAML configuration string for a mode.

        Args:
            mode: Timing mode

        Returns:
            Example YAML configuration as string
        """
        config = cls.get_default_config(mode)

        yaml_lines = ["timing_config:"]
        for key, value in config.items():
            if isinstance(value, str):
                yaml_lines.append(f'  {key}: "{value}"')
            elif value is None:
                yaml_lines.append(f"  {key}: null")
            elif isinstance(value, dict):
                yaml_lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    yaml_lines.append(f'    {sub_key}: "{sub_value}"')
            else:
                yaml_lines.append(f"  {key}: {value}")

        return "\n".join(yaml_lines)
