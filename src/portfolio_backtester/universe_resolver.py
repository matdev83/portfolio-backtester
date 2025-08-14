import logging
from typing import Any, Dict, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)

# ---------------- Internal helpers to reduce complexity ----------------


def _normalize_ticker(t: Any) -> str:
    if not isinstance(t, str):
        raise ValueError(f"All tickers must be strings, got: {type(t)}")
    nt = t.strip().upper()
    if not nt:
        raise ValueError("Empty ticker found in tickers list")
    return nt


def _resolve_single_symbol(config: Dict[str, Any]) -> List[str]:
    ticker = config.get("params", {}).get("ticker") or config.get("ticker")
    if not ticker:
        raise ValueError("single_symbol universe requires 'ticker' field")
    return [_normalize_ticker(ticker)]


def _resolve_method(config: Dict[str, Any], current_date: Optional[pd.Timestamp]) -> List[str]:
    method_name = config.get("method_name", "get_top_weight_sp500_components")
    if method_name == "get_top_weight_sp500_components":
        from .universe import get_top_weight_sp500_components

        if current_date is None:
            current_date = pd.Timestamp.now().normalize()
        n_holdings = config.get("n_holdings", 50)
        exact = config.get("exact", False)
        try:
            tickers = get_top_weight_sp500_components(date=current_date, n=n_holdings, exact=exact)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Loaded {len(tickers)} tickers using method '{method_name}'")
            return tickers
        except Exception as e:
            raise ValueError(f"Failed to get universe using method '{method_name}': {e}")
    raise ValueError(f"Unknown universe method: {method_name}")


def _resolve_fixed(config: Dict[str, Any]) -> List[str]:
    tickers = config.get("tickers", [])
    if not isinstance(tickers, list):
        raise ValueError("universe_config.tickers must be a list")
    if not tickers:
        raise ValueError("universe_config.tickers cannot be empty")
    normalized = [_normalize_ticker(t) for t in tickers]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Loaded {len(normalized)} tickers from fixed list")
    return normalized


def _resolve_named(config: Dict[str, Any]) -> List[str]:
    from .universe_loader import (
        load_named_universe,
        load_multiple_named_universes,
        UniverseLoaderError,
    )

    universe_name = config.get("universe_name")
    universe_names = config.get("universe_names")
    if universe_name and universe_names:
        raise ValueError("Cannot specify both 'universe_name' and 'universe_names'")
    if not universe_name and not universe_names:
        raise ValueError(
            "Must specify either 'universe_name' or 'universe_names' for named universe type"
        )
    try:
        if universe_name:
            tickers: List[str] = load_named_universe(universe_name)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Loaded named universe '{universe_name}' with {len(tickers)} tickers")
            return tickers
        if not isinstance(universe_names, list):
            raise ValueError("universe_names must be a list")
        if not universe_names:
            raise ValueError("universe_names cannot be empty")
        tickers = load_multiple_named_universes(universe_names)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Loaded {len(universe_names)} named universes with {len(tickers)} unique tickers"
            )
        return tickers
    except UniverseLoaderError as e:
        raise ValueError(f"Failed to load named universe: {e}")


# Internal function used by universe providers
def resolve_universe_config(
    universe_config: dict, current_date: Optional[pd.Timestamp] = None
) -> list[str]:
    """
    Resolve universe configuration to a list of tickers.

    Args:
        universe_config: Universe configuration dictionary
        current_date: Optional current date for universe resolution

    Returns:
        List of ticker symbols

    Raises:
        ValueError: If universe_config is invalid or cannot be resolved
    """
    # Support legacy shorthand for single_symbol universe
    if (
        "name" in universe_config and universe_config.get("name") == "single_symbol"
    ) or universe_config.get("type") == "single_symbol":
        return _resolve_single_symbol(universe_config)

    universe_type = universe_config.get("type")

    if not universe_type:
        raise ValueError("universe_config must specify 'type'")

    if universe_type == "method":
        return _resolve_method(universe_config, current_date)

    elif universe_type == "fixed":
        return _resolve_fixed(universe_config)

    elif universe_type == "named":
        return _resolve_named(universe_config)

    else:
        raise ValueError(f"Unknown universe type: {universe_type}")
