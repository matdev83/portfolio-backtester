import sys, pprint
from edgar import Company, set_identity

set_identity("mateusz@bartczak.me")          # your e-mail / UA

if len(sys.argv) != 2:
    sys.exit("Usage: python inspect_filing.py <accession_no>")

acc = sys.argv[1]                            # e.g. 0001752724-25-125179
fund = Company("0000884394")                 # SPY
filing = next(f for f in fund.get_filings() if f.accession_no == acc)

obj = filing.obj()

print("OBJ TYPE:", type(obj))
print("obj attrs:", [a for a in dir(obj) if not a.startswith("_")])

if hasattr(obj, "portfolio"):
    port = obj.portfolio
    print("\nPORTFOLIO TYPE:", type(port))
    print("portfolio attrs:", [a for a in dir(port) if not a.startswith('_')])
    if hasattr(port, "holdings"):
        print("\nholdings length:", len(port.holdings))
        pprint.pp(port.holdings[0], width=120)