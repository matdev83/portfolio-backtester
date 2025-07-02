import pytest
pytestmark = pytest.mark.network
pytest.skip("network test", allow_module_level=True)

import datetime as dt
from portfolio_backtester import spy_holdings

print(spy_holdings.ssga_daily(dt.date(2024,6,28)))
