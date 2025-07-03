import subprocess
import datetime

def run_recovery():
    today = datetime.date.today()
    # Assuming the script was last run with --end 2025-06-30 or similar
    # The --update flag will make the script figure out the actual start date from cache
    command = [
        "python",
        "-m",
        "src.portfolio_backtester.spy_holdings",
        "--update",
        "--end",
        today.strftime("%Y-%m-%d"),
        "--out",
        "spy_holdings_full.parquet",
        "--log-level",
        "INFO"
    ]
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print("Recovery script finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Recovery script failed with error: {e}")

if __name__ == "__main__":
    run_recovery()
