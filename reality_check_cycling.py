from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------
# CONFIG
# ------------------
BASE_DIR = Path(__file__).resolve().parent
IN_PATH = BASE_DIR / "out" / "workouts_enriched.csv"
OUT_PATH = BASE_DIR / "out" / "cycling_final_reality_check_from_pipeline.csv"


def to_miles(distance: float, unit: str | None) -> float:
    if pd.isna(distance):
        return np.nan
    u = (unit or "").strip().lower()
    if u in ("mi", "mile", "miles"):
        return float(distance)
    if u in ("km", "kilometer", "kilometers"):
        return float(distance) * 0.621371
    if u in ("m", "meter", "meters"):
        return float(distance) / 1609.344
    # unknown unit; leave as-is
    return float(distance)


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}\n"
            "Run your pipeline first so out/workouts_enriched.csv exists."
        )

    df = pd.read_csv(IN_PATH, parse_dates=["startDate", "endDate"], low_memory=False)

    # normalize tz if present
    if hasattr(df["startDate"].dtype, "tz") and df["startDate"].dtype.tz is not None:
        df["startDate"] = df["startDate"].dt.tz_localize(None)

    # filter cycling
    if "workoutActivityType" not in df.columns:
        raise ValueError("workouts_enriched.csv is missing 'workoutActivityType'")

    cyc = df[df["workoutActivityType"] == "HKWorkoutActivityTypeCycling"].copy()
    if cyc.empty:
        print("No cycling workouts found.")
        return

    # duration_min safety
    if "duration_min" not in cyc.columns:
        if "duration" in cyc.columns:
            cyc["duration_min"] = pd.to_numeric(cyc["duration"], errors="coerce")
        else:
            cyc["duration_min"] = np.nan

    # distance fields safety (depends on whether your pipeline exports WorkoutStatistics)
    if "distance_cycling" not in cyc.columns:
        cyc["distance_cycling"] = np.nan
    if "distance_cycling_unit" not in cyc.columns:
        cyc["distance_cycling_unit"] = None

    # convert distance to miles
    cyc["distance_mi"] = [
        to_miles(d, u) for d, u in zip(cyc["distance_cycling"], cyc["distance_cycling_unit"])
    ]

    # build requested display columns
    cyc["date"] = cyc["startDate"].dt.date.astype(str)
    cyc["start_time"] = cyc["startDate"].dt.strftime("%H:%M")

    # ensure HR columns exist
    for col in ["avg_hr", "max_hr", "hr_samples"]:
        if col not in cyc.columns:
            cyc[col] = np.nan

    out = cyc[[
        "date",
        "start_time",
        "distance_mi",
        "duration_min",
        "avg_hr",
        "max_hr",
        "hr_samples",
    ]].copy()

    out = out.sort_values(["date", "start_time"])

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}\n")

    print("Preview (most recent 40 rides):")
    print(out.tail(40).to_string(index=False))


if __name__ == "__main__":
    main()