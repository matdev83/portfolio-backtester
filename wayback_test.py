import datetime as dt
from portfolio_backtester import spy_holdings

dt_target = dt.date.today()
print("Fetching", dt_target)
res = spy_holdings.ssga_daily(dt_target)
print("Returned:", None if res is None else len(res))
if res is not None:
    print(res.head()) 