from __future__ import annotations

import csv
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    grid = _REPO / "dev/spy_seasonal_subset_exhaustive_legacy_sortino_2005_2024_hd10/all_subsets_metrics.csv"
    summary = _REPO / "data/reports/seasonal_month_spylong_sortino_summary.csv"
    by_mask: dict[int, dict[str, str]] = {}
    with grid.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_mask[int(row["bitmask"])] = row
    rows_sum = list(csv.DictReader(summary.open(encoding="utf-8")))
    assert len(rows_sum) == 12
    print("single_month_legacy_sortino_vs_csv_strategy_Sortino")
    print(f"{'month':<12} {'mask':>5} {'grid_S':>14} {'csv_S':>14} {'delta':>10}")
    for i, r in enumerate(rows_sum):
        mname = r["month"]
        mask = 1 << i
        g = float(by_mask[mask]["Sortino"])
        c = float(r["strategy_Sortino"])
        print(f"{mname:<12} {mask:5d} {g:14.9f} {c:14.9f} {g-c:10.6f}")
    jul = rows_sum[6]
    mask_jul = 1 << 6
    print("\njuly_total_return_grid_vs_csv")
    print("grid TR", by_mask[mask_jul]["Total Return"])
    print("csv TR ", jul["strategy_Total_Return"])


if __name__ == "__main__":
    main()
