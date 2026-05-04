"""Default provider wiring for :class:`BaseStrategy`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Union

if TYPE_CHECKING:
    from ....canonical_config import CanonicalScenarioConfig


def build_default_strategy_providers(
    provider_init_arg: Union[Mapping[str, Any], "CanonicalScenarioConfig"],
) -> tuple[Any, Any, Any, Any, Any]:
    """Construct universe, sizing, stop-loss, take-profit, and risk-off providers.

    Args:
        provider_init_arg: Canonical scenario or raw strategy parameter mapping.

    Returns:
        Tuple ``(universe_provider, position_sizer_provider, stop_loss_provider,
        take_profit_provider, risk_off_signal_provider)``.
    """
    from ....interfaces.universe_provider_interface import UniverseProviderFactory
    from ....interfaces.position_sizer_provider_interface import PositionSizerProviderFactory
    from ....interfaces.stop_loss_provider_interface import StopLossProviderFactory
    from ....interfaces.take_profit_provider_interface import TakeProfitProviderFactory
    from ....risk_off_signals import RiskOffSignalProviderFactory

    universe_provider = UniverseProviderFactory.create_config_provider(provider_init_arg)
    position_sizer_provider = PositionSizerProviderFactory.get_default_provider(provider_init_arg)
    stop_loss_provider = StopLossProviderFactory.get_default_provider(provider_init_arg)
    take_profit_provider = TakeProfitProviderFactory.get_default_provider(provider_init_arg)
    risk_off_signal_provider = RiskOffSignalProviderFactory.get_default_provider(provider_init_arg)
    return (
        universe_provider,
        position_sizer_provider,
        stop_loss_provider,
        take_profit_provider,
        risk_off_signal_provider,
    )
