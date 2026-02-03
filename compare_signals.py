import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def compare_signals():
    print("Fetching data...")
    # Fetch SPY and VIX
    spy = yf.download("SPY", start="2012-01-01", end="2023-12-31")
    vix = yf.download("^VIX", start="2012-01-01", end="2023-12-31")
    
    if spy.empty or vix.empty:
        print("Failed to fetch data.")
        return

    # Standardize column index if they are multi-index (recent yfinance change)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # 1. SPX 200-day SMA (approx 40 weeks)
    # The "BenchmarkWeeklySma" logic uses weekly closes
    spy_weekly = spy["Close"].resample("W").last()
    spy_sma_200 = spy_weekly.rolling(window=40).mean()
    
    # Broadcast weekly signal back to daily index
    spy_sma_daily = spy_sma_200.reindex(spy.index, method="ffill")
    spy_signal = (spy["Close"] < spy_sma_daily)
    
    # 2. VIX 20-day SMA
    vix_sma_20 = vix["Close"].rolling(window=20).mean()
    # Reindex to match SPY dates perfectly
    vix_sma_daily = vix_sma_20.reindex(spy.index)
    vix_signal = (vix_sma_daily > 21)
    
    # Create combined dataframe
    df = pd.DataFrame(index=spy.index)
    df["SPY_Price"] = spy["Close"]
    df["SPX_200SMA_RO"] = spy_signal.astype(int)
    df["VIX_20SMA_RO"] = vix_signal.astype(int)
    
    # Statistics
    print(f"Total days: {len(df)}")
    print(f"SPX 200 SMA Cash %: {df['SPX_200SMA_RO'].mean():.2%}")
    print(f"VIX 20 SMA Cash %: {df['VIX_20SMA_RO'].mean():.2%}")
    
    # Agreement: Both say cash or both say risk-on
    agreement = (df["SPX_200SMA_RO"] == df["VIX_20SMA_RO"]).mean()
    print(f"Signal Agreement: {agreement:.2%}")
    
    # Calculate returns while in market
    daily_rets = df["SPY_Price"].pct_change()
    spy_sma_rets = daily_rets * (1 - df["SPX_200SMA_RO"].shift(1))
    vix_sma_rets = daily_rets * (1 - df["VIX_20SMA_RO"].shift(1))
    
    spy_sma_cum = (1 + spy_sma_rets.fillna(0)).cumprod()
    vix_sma_cum = (1 + vix_sma_rets.fillna(0)).cumprod()
    spy_bh_cum = (1 + daily_rets.fillna(0)).cumprod()
    
    print(f"SPY B&H Final: {spy_bh_cum.iloc[-1]:.2f}")
    print(f"SPX SMA Strategy Final: {spy_sma_cum.iloc[-1]:.2f}")
    print(f"VIX SMA Strategy Final: {vix_sma_cum.iloc[-1]:.2f}")

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(spy_bh_cum.index, spy_bh_cum, label="SPY B&H", color='gray', alpha=0.5)
    plt.plot(spy_sma_cum.index, spy_sma_cum, label="SPX 200 SMA System", color='blue')
    plt.plot(vix_sma_cum.index, vix_sma_cum, label="VIX 20 SMA System", color='red')
    plt.title("Growth of $1: Risk-Off Signal Comparison")
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 1, 2)
    plt.fill_between(df.index, 0, df["SPX_200SMA_RO"], color="blue", alpha=0.3, label="SPX SMA RO (1=Cash)")
    plt.fill_between(df.index, 0, df["VIX_20SMA_RO"], color="red", alpha=0.3, label="VIX SMA RO (1=Cash)")
    plt.title("Cash Allocation (Risk-Off Signal)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("signal_comparison.png")
    print("Plot saved as signal_comparison.png")

if __name__ == "__main__":
    compare_signals()
