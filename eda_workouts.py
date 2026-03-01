import pandas as pd

workouts = pd.read_csv("out/workouts.csv", parse_dates=["startDate", "endDate"], low_memory=False)

workouts["week"] = workouts["startDate"].dt.tz_localize(None).dt.to_period("W").dt.start_time

import numpy as np

workouts["duration"] = pd.to_numeric(workouts["duration"], errors="coerce")

# Map durationUnit -> multiplier to minutes
unit_to_min = {
    "sec": 1/60,
    "s": 1/60,
    "min": 1,
    "hr": 60,
    "h": 60
}

workouts["duration_min"] = workouts.apply(
    lambda r: r["duration"] * unit_to_min.get(str(r.get("durationUnit")).lower(), np.nan),
    axis=1
)

# If unit missing, assume seconds (common), but only fill where duration_min is NaN
workouts.loc[workouts["duration_min"].isna() & workouts["duration"].notna(), "duration_min"] = workouts["duration"] / 60

cols_wanted = ["workoutActivityType", "duration_min", "totalDistance", "totalEnergyBurned"]
cols_present = [c for c in cols_wanted if c in workouts.columns]

print("Present columns:", cols_present)
print(workouts[cols_present].tail(10))

# workout counts by type
counts = workouts["workoutActivityType"].value_counts().head(15)
print(counts)

# weekly minutes (good first training-volume chart later)
workouts["week"] = workouts["startDate"].dt.to_period("W").dt.start_time
weekly_minutes = workouts.groupby("week")["duration_min"].sum().sort_index()
print(weekly_minutes.tail(12))

print(workouts["durationUnit"].value_counts(dropna=False).head(10))
print(workouts[["duration", "durationUnit", "duration_min"]].head(5))

