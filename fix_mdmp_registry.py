
import json
import logging
from pathlib import Path
from market_data_multi_provider.sp500 import builder
from market_data_multi_provider.sp500.config import clean_ticker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_registry():
    # 1. Load History
    logger.info("Loading S&P 500 history...")
    builder._ensure_history_loaded()
    if builder._HISTORY_DF is None or builder._HISTORY_DF.empty:
        logger.error("Failed to load S&P 500 history.")
        return

    unique_tickers_raw = builder._HISTORY_DF['ticker'].unique()
    unique_tickers = sorted([t for t in unique_tickers_raw if t is not None])
    logger.info(f"Found {len(unique_tickers)} unique tickers in history.")

    # 2. Load symbols.json
    # Assuming relative path to MDMP repo based on user info
    mdmp_repo_path = Path(r"c:\Users\Mateusz\source\repos\market-data-multi-provider")
    symbols_path = mdmp_repo_path / "src" / "market_data_multi_provider" / "resources" / "symbols.json"
    
    if not symbols_path.exists():
        logger.error(f"symbols.json not found at {symbols_path}")
        return

    with open(symbols_path, 'r') as f:
        data = json.load(f)

    current_symbols = data['symbols']
    
    # 3. Build lookup for existing aliases and IDs
    existing_map = set()
    for sid, spec in current_symbols.items():
        existing_map.add(sid)
        for alias in spec.get('aliases', []):
            existing_map.add(alias)
            existing_map.add(f"NYSE:{alias}") # implicitly handled
            existing_map.add(f"NASDAQ:{alias}") # implicitly handled

    # Known NASDAQ tickers (extended list)
    nasdaq_set = {
        "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "AMGN", "AVGO", 
        "CMCSA", "COST", "CSCO", "HON", "INTC", "INTU", "NFLX", "PEP", "QCOM", "TXN", "AMD", 
        "ANSS", "ASML", "AZN", "BIIB", "BKNG", "BMRN", "CDNS", "CHTR", "CPRT", "CRWD", "CSX", 
        "CTAS", "CTSH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "FAST", "FISV", "FTNT", "GILD", 
        "IDXX", "ILMN", "ISRG", "JD", "KDP", "KHC", "KLAC", "LCID", "LRCX", "LULU", "MAR", 
        "MCHP", "MDLZ", "MELI", "MNST", "MRNA", "MRVL", "MU", "NXPI", "ODFL", "ORLY", "PANW", 
        "PAYX", "PCAR", "PDD", "PYPL", "REGN", "ROST", "SBUX", "SGEN", "SIRI", "SNPS", "SPLK", 
        "SWKS", "TEAM", "TMUS", "VRSK", "VRSN", "VRTX", "WBA", "WDAY", "XEL", "ZM", "ZS",
        "ADSK", "ALGN", "AMAT", "CHKP", "CTXS", "DOCU", "EBAY", "FOX", "FOXA", "GRMN", "HSIC",
        "INCY", "KDP", "LBRDK", "LBRDA", "MRAY", "MTCH", "NTAP", "NTES", "OKTA", "PTON", "ROKU",
        "SWKS", "SIRI", "SPLK", "TCOM", "VRSK", "VRSN", "VRTX", "WDC", "WYNN", "XLNX", "ZM"
    }

    added_count = 0
    for t in unique_tickers:
        t_clean = clean_ticker(t)
        if not t_clean: 
            continue
            
        # Heuristic check if already present
        # We check exact match on Alias or ID suffix
        is_present = False
        for sid in current_symbols:
            if sid.endswith(f":{t_clean}"):
                is_present = True
                break
        
        if is_present:
            continue

        # If not present, add it
        # Exchange guess
        exch = "NASDAQ" if t_clean in nasdaq_set else "NYSE"
        
        sid = f"{exch}:{t_clean}"
        
        # Stooq symbol: usually lowercase.us
        stooq_sym = f"{t_clean.lower()}.us"
        
        # YFinance symbol: usually just ticker
        yf_sym = t_clean
        
        # entry
        entry = {
            "preferred_provider": "stooq",
            "calendar_id": "NYSE",
            "description": f"S&P 500 Component {t_clean}",
            "providers": {
                "stooq": stooq_sym,
                "yfinance": yf_sym,
                "tradingview": f"{exch}:{t_clean}"
            },
            "aliases": [t_clean]
        }
        
        current_symbols[sid] = entry
        added_count += 1
        
    logger.info(f"Adding {added_count} new symbols to registry.")
    
    if added_count > 0:
        with open(symbols_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Registry updated successfully.")
    else:
        logger.info("No new symbols to add.")

if __name__ == "__main__":
    fix_registry()
