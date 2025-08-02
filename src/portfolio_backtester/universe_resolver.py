import logging
import pandas as pd

logger = logging.getLogger(__name__)

def resolve_universe_config(universe_config: dict, current_date: pd.Timestamp = None) -> list[str]:
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
    from .universe_loader import (
        load_named_universe, 
        load_multiple_named_universes,
        UniverseLoaderError
    )
    
    # Support legacy shorthand for single_ticker universe
    # Handle single_ticker shorthand variations
    if (
        ('name' in universe_config and universe_config.get('name') == 'single_ticker')
        or universe_config.get('type') == 'single_ticker'
    ):
        ticker = (
            universe_config.get('params', {}).get('ticker')
            or universe_config.get('ticker')
        )
        if not ticker:
            raise ValueError("single_ticker universe requires 'ticker' field")
        return [str(ticker).strip().upper()]

    universe_type = universe_config.get("type")

    if not universe_type:
        raise ValueError("universe_config must specify 'type'")
    
    if universe_type == "method":
        method_name = universe_config.get("method_name", "get_top_weight_sp500_components")
        
        if method_name == "get_top_weight_sp500_components":
            from .universe import get_top_weight_sp500_components
            
            if current_date is None:
                current_date = pd.Timestamp.now().normalize()
            
            n_holdings = universe_config.get("n_holdings", 50)
            exact = universe_config.get("exact", False)
            
            try:
                tickers = get_top_weight_sp500_components(
                    date=current_date,
                    n=n_holdings,
                    exact=exact
                )
                logger.info(f"Loaded {len(tickers)} tickers using method '{method_name}'")
                return tickers
            except Exception as e:
                raise ValueError(f"Failed to get universe using method '{method_name}': {e}")
        else:
            raise ValueError(f"Unknown universe method: {method_name}")
    
    elif universe_type == "fixed":
        tickers = universe_config.get("tickers", [])
        if not isinstance(tickers, list):
            raise ValueError("universe_config.tickers must be a list")
        
        if not tickers:
            raise ValueError("universe_config.tickers cannot be empty")
        
        normalized_tickers = []
        for ticker in tickers:
            if not isinstance(ticker, str):
                raise ValueError(f"All tickers must be strings, got: {type(ticker)}")
            normalized_ticker = ticker.strip().upper()
            if not normalized_ticker:
                raise ValueError("Empty ticker found in tickers list")
            normalized_tickers.append(normalized_ticker)
        
        logger.info(f"Loaded {len(normalized_tickers)} tickers from fixed list")
        return normalized_tickers
    
    elif universe_type == "named":
        universe_name = universe_config.get("universe_name")
        universe_names = universe_config.get("universe_names")
        
        if universe_name and universe_names:
            raise ValueError("Cannot specify both 'universe_name' and 'universe_names'")
        
        if not universe_name and not universe_names:
            raise ValueError("Must specify either 'universe_name' or 'universe_names' for named universe type")
        
        try:
            if universe_name:
                tickers = load_named_universe(universe_name)
                logger.info(f"Loaded named universe '{universe_name}' with {len(tickers)} tickers")
                return tickers
            else:
                if not isinstance(universe_names, list):
                    raise ValueError("universe_names must be a list")
                
                if not universe_names:
                    raise ValueError("universe_names cannot be empty")
                
                tickers = load_multiple_named_universes(universe_names)
                logger.info(f"Loaded {len(universe_names)} named universes with {len(tickers)} unique tickers")
                return tickers
                
        except UniverseLoaderError as e:
            raise ValueError(f"Failed to load named universe: {e}")
    
    else:
        raise ValueError(f"Unknown universe type: {universe_type}")
