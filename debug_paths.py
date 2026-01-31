import portfolio_backtester
import portfolio_backtester.strategies as strategies
print(f"portfolio_backtester file: {portfolio_backtester.__file__}")
print(f"strategies file: {strategies.__file__}")

try:
    import src.portfolio_backtester as src_pb
    print(f"src.portfolio_backtester file: {src_pb.__file__}")
except ImportError:
    print("src.portfolio_backtester NOT found")
