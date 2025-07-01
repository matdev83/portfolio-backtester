import sys, pprint
from edgar import Filing, set_identity

set_identity("mateusz@bartczak.me")          # your e-mail for SEC

if len(sys.argv) != 2:
    sys.exit("Usage: python inspect_filing.py <accession_no>")

asc = sys.argv[1]                            # e.g. 0001752724-25-125179
filing = Filing.from_accession(asc)
obj = filing.obj()

print("OBJ TYPE:", type(obj))
print("obj attrs:", [a for a in dir(obj) if not a.startswith("_")])

if hasattr(obj, "portfolio"):
    port = obj.portfolio
    print("\nPORTFOLIO TYPE:", type(port))
    print("portfolio attrs:", [a for a in dir(port) if not a.startswith("_")])
    if hasattr(port, "holdings"):
        print("\nholdings length:", len(port.holdings))
        pprint.pp(port.holdings[0], width=120)
else:
    print('obj attributes:', dir(obj)[:40]) 