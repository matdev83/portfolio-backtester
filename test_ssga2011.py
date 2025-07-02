import pytest
pytestmark = pytest.mark.network
pytest.skip("network test", allow_module_level=True)

import datetime as dt
from portfolio_backtester import spy_holdings

date = dt.date(2011, 4, 1)
df = spy_holdings.ssga_daily(date)
print('None returned' if df is None else df.head())
print('Rows:', 0 if df is None else len(df))
