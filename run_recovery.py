import subprocess
import datetime
import os

def run_recovery():
    today = datetime.date.today()
    venv_python = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".venv",
        "Scripts",
        "python.exe"
    )

    output_file = "spy_holdings_full.parquet"
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        output_file
    )

    # Step 1: Rebuild the full history from existing cache files
    print("Attempting to rebuild full history from cache...")
    rebuild_command = [
        venv_python,
        "-m",
        "src.portfolio_backtester.spy_holdings",
        "--rebuild",
        "--start",
        "2004-01-01", # Start from the earliest possible date for a full rebuild
        "--end",
        today.strftime("%Y-%m-%d"),
        "--out",
        output_file,
        "--log-level",
        "INFO"
    ]
    print(f"Running rebuild command: {' '.join(rebuild_command)}")
    try:
        subprocess.run(rebuild_command, check=True)
        print("Rebuild finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Rebuild failed with error: {e}")
        return # Exit if rebuild fails

    # Step 2: Once rebuilt, perform an update to catch any new data
    print("Attempting to update history...")
    update_command = [
        venv_python,
        "-m",
        "src.portfolio_backtester.spy_holdings",
        "--update",
        "--end",
        today.strftime("%Y-%m-%d"),
        "--out",
        output_file,
        "--log-level",
        "INFO"
    ]
    print(f"Running update command: {' '.join(update_command)}")
    try:
        subprocess.run(update_command, check=True)
        print("Update finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Update failed with error: {e}")

if __name__ == "__main__":
    run_recovery()
