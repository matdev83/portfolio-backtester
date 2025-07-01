from portfolio_backtester.cusip_mapping import CusipMappingDB

db = CusipMappingDB()
print(db._lookup_duckduckgo('FB', throttle=0)) 