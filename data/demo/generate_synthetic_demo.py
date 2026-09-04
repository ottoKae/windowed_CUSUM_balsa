"""Generate a small, deterministic, non-geographic Windowed CUSUM demo."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path


OUT = Path(__file__).resolve().parent


def add_months(start: date, offset: int) -> date:
    month_index = start.year * 12 + start.month - 1 + offset
    return date(month_index // 12, month_index % 12 + 1, 1)


def values(stand: int, month: int) -> tuple[str, str]:
    seasonal = 0.35 * math.sin(2 * math.pi * (month % 12) / 12)
    if stand == 1:
        # Invented abrupt disturbance at month 30 followed by gradual recovery.
        effect = 0.0 if month < 30 else -5.0 + min(month - 30, 24) * 0.16
    else:
        effect = 0.15 * math.cos(2 * math.pi * (month % 18) / 18)
    vv = -8.4 + seasonal + effect
    vh = -15.2 + 0.8 * seasonal + 0.82 * effect
    if (stand, month) in {(1, 8), (1, 44), (2, 21)}:
        return "", ""
    return f"{vv:.3f}", f"{vh:.3f}"


def main() -> None:
    monthly_path = OUT / "monthly_s1_synthetic.csv"
    with monthly_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["ID", "time", "VV_dB", "VH_dB"])
        for stand in (1, 2):
            for month in range(60):
                vv, vh = values(stand, month)
                writer.writerow(
                    [f"DEMO_STAND_{stand:03d}", add_months(date(2017, 1, 1), month), vv, vh]
                )

    reference_path = OUT / "reference_events_synthetic.csv"
    with reference_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["ID", "event_month", "reference_type"])
        writer.writerow(["DEMO_STAND_001", "2019-07-01", "synthetic"])


if __name__ == "__main__":
    main()

