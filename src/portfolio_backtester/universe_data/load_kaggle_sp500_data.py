"""Utility script to convert Kaggle SP-500 CSV weights into a Parquet file.

This is **not** meant to run on import.  The heavy I/O is guarded by the
``if __name__ == "__main__"`` block so the portfolio_backtester package can be
imported without side-effects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = repo_root / "data" / "sp500_historical.csv"
    out_dir = repo_root / "data" / "kaggle_sp500_weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "sp500_historical.parquet"
    return csv_path, parquet_path


def main() -> None:  # pragma: no cover – dev helper
    csv_file_path, output_parquet_path = _default_paths()

    print(f"Loading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    print("Converting 'date' column to datetime…")
    df["date"] = pd.to_datetime(df["date"])

    print(f"Saving processed data to: {output_parquet_path}")
    df.to_parquet(output_parquet_path, index=False)

    print("Data loading complete.")
    print("\n--- Info of saved Parquet file ---")
    loaded_df = pd.read_parquet(output_parquet_path)
    loaded_df.info()


if __name__ == "__main__":
    main()
