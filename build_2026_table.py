from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------
# CONFIG
# ------------------
BASE_DIR = Path(__file__).resolve().parent
IN_PATH = BASE_DIR / "out" / "workouts_enriched.csv"
OUT_PATH = BASE_DIR / "out" / "build_2026_daily_table.csv"

BUILD_START = pd.Timestamp("2026-01-07")
BUILD_END = None  # set to pd.Timestamp("2026-06-07") if you want a hard cap


# ------------------
# Helpers
# ------------------
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
    return float(distance)


def clean_activity(workout_activity_type: str) -> str:
    if pd.isna(workout_activity_type):
        return "Unknown"
    return str(workout_activity_type).replace("HKWorkoutActivityType", "")


def pick_distance(row: pd.Series) -> tuple[float, str]:
    """
    Choose the most relevant distance field if present.
    Priority:
      1) distance_cycling
      2) distance_running
      3) distance_swimming
    Returns: (distance_miles, 'mi')
    """
    # cycling
    if "distance_cycling" in row and pd.notna(row["distance_cycling"]):
        return to_miles(row["distance_cycling"], row.get("distance_cycling_unit")), "mi"

    # running/walking
    if "distance_running" in row and pd.notna(row["distance_running"]):
        return to_miles(row["distance_running"], row.get("distance_running_unit")), "mi"

    # swimming (if you later add it)
    if "distance_swimming" in row and pd.notna(row["distance_swimming"]):
        return to_miles(row["distance_swimming"], row.get("distance_swimming_unit")), "mi"

    return np.nan, ""


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run your pipeline first.")

    df = pd.read_csv(IN_PATH, parse_dates=["startDate", "endDate"], low_memory=False)

    # normalize tz if present
    if hasattr(df["startDate"].dtype, "tz") and df["startDate"].dtype.tz is not None:
        df["startDate"] = df["startDate"].dt.tz_localize(None)

    # filter to 2026 build window
    sub = df[df["startDate"] >= BUILD_START].copy()
    if BUILD_END is not None:
        sub = sub[sub["startDate"] <= BUILD_END]

    if sub.empty:
        print("No workouts found in the 2026 build window.")
        return

    # duration_min safety
    if "duration_min" not in sub.columns:
        sub["duration_min"] = pd.to_numeric(sub.get("duration"), errors="coerce")

    # Build table fields
    sub["day"] = sub["startDate"].dt.strftime("%a")
    sub["date"] = sub["startDate"].dt.date.astype(str)
    sub["event"] = sub["workoutActivityType"].apply(clean_activity)

    # distance (event-specific)
    dist_vals = sub.apply(pick_distance, axis=1, result_type="expand")
    sub["distance"] = dist_vals[0]
    sub["distance_unit"] = dist_vals[1]

    # avg_hr safety
    if "avg_hr" not in sub.columns:
        sub["avg_hr"] = np.nan

    out = sub[["day", "date", "event", "distance", "distance_unit", "duration_min", "avg_hr"]].copy()

    # Optional: round for readability
    out["distance"] = out["distance"].round(2)
    out["duration_min"] = out["duration_min"].round(1)
    out["avg_hr"] = out["avg_hr"].round(1)

    out = out.sort_values(["date", "day", "event"])

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}\n")
    print("Preview (most recent 25 rows):")
    print(out.tail(25).to_string(index=False))


if __name__ == "__main__":
    main()