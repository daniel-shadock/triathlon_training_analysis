import pandas as pd
import numpy as np

# ---------- Load ----------
workouts = pd.read_csv(
    "out/workouts.csv",
    parse_dates=["startDate", "endDate", "creationDate"],
    low_memory=False
)

hr = pd.read_csv(
    "out/heart_rate.csv",
    parse_dates=["startDate", "endDate", "creationDate"],
    low_memory=False
)

# ---------- Clean / normalize ----------
# duration_min: assume durationUnit is minutes in your export (matches your earlier results)
workouts["duration"] = pd.to_numeric(workouts["duration"], errors="coerce")
workouts["duration_min"] = workouts["duration"]

# friendly activity labels
workouts["activity"] = workouts["workoutActivityType"].str.replace("HKWorkoutActivityType", "", regex=False)

# remove tz for consistent comparisons & weekly buckets
workouts["startDate"] = workouts["startDate"].dt.tz_localize(None)
workouts["endDate"] = workouts["endDate"].dt.tz_localize(None)
hr["startDate"] = hr["startDate"].dt.tz_localize(None)

hr["value"] = pd.to_numeric(hr["value"], errors="coerce")

# ---------- Attach HR samples to workouts efficiently ----------
# We'll assign each HR sample to the workout window it belongs to using searchsorted.
w = workouts.reset_index().rename(columns={"index": "workout_id"}).sort_values("startDate")
r = hr.sort_values("startDate")

w_starts = w["startDate"].to_numpy()
w_ends = w["endDate"].to_numpy()
r_times = r["startDate"].to_numpy()

idx = np.searchsorted(w_starts, r_times, side="right") - 1
valid = idx >= 0

workout_id = np.full(len(r), np.nan)
candidate_pos = idx[valid]

# keep only HR samples that fall before the candidate workout's end
within = r_times[valid] <= w_ends[candidate_pos]
workout_id_valid = w.iloc[candidate_pos[within]]["workout_id"].to_numpy()

workout_id[np.where(valid)[0][within]] = workout_id_valid
r["workout_id"] = workout_id

# ---------- HR summary per workout ----------
hr_summary = (
    r.dropna(subset=["workout_id"])
     .groupby("workout_id")["value"]
     .agg(avg_hr="mean", max_hr="max")
     .reset_index()
)

df = w.merge(hr_summary, on="workout_id", how="left")

# ---------- Define windows ----------
windows = [
    ("Build_to_2024-09-28", "2024-04-28", "2024-09-28"),
    ("Build_to_2025-07-13", "2025-02-13", "2025-07-13"),
    ("Progress_since_2026-01-07", "2026-01-07", None),  # None = through latest data
]

# ---------- Helper: TRIMP-like HR load proxy ----------
# Since we don't have HRrest/HRmax in the file, we estimate intensity from avg_hr relative to your own distribution.
# This is still very useful for comparing build blocks.
global_hr = df["avg_hr"].dropna()
hr_p10, hr_p90 = np.nanpercentile(global_hr, [10, 90]) if len(global_hr) else (np.nan, np.nan)

def hr_intensity(avg_hr):
    # scale roughly 0..1 based on your own easy-to-hard range
    if np.isnan(avg_hr) or np.isnan(hr_p10) or np.isnan(hr_p90) or hr_p90 == hr_p10:
        return np.nan
    return np.clip((avg_hr - hr_p10) / (hr_p90 - hr_p10), 0, 1)

df["intensity"] = df["avg_hr"].apply(hr_intensity)
df["load"] = (df["duration_min"] * df["intensity"]).astype("float")  # minutes * intensity

# ---------- Summaries ----------
def summarize(block_name, start, end):
    sub = df[df["startDate"] >= pd.Timestamp(start)].copy()
    if end is not None:
        sub = sub[sub["startDate"] <= pd.Timestamp(end)]
    else:
        end = sub["startDate"].max().date().isoformat() if len(sub) else "NA"

    if len(sub) == 0:
        return None, None, None

    # weekly bucket
    sub["week"] = sub["startDate"].dt.to_period("W").dt.start_time

    overall = pd.DataFrame([{
        "block": block_name,
        "start": start,
        "end": end,
        "workouts": len(sub),
        "total_hours": sub["duration_min"].sum() / 60,
        "hours_per_week": (sub.groupby("week")["duration_min"].sum().mean() / 60),
        "avg_workout_min": sub["duration_min"].mean(),
        "avg_hr": sub["avg_hr"].mean(),
        "max_hr": sub["max_hr"].mean(),  # mean of per-workout max HR
        "load_total": sub["load"].sum(),
        "load_per_week": sub.groupby("week")["load"].sum().mean(),
    }])

    by_activity = (
        sub.groupby("activity")
           .agg(
                workouts=("workout_id", "count"),
                total_hours=("duration_min", lambda x: x.sum()/60),
                hours_per_week=("duration_min", lambda x: (x.groupby(sub.loc[x.index, "week"]).sum().mean()/60)),
                avg_workout_min=("duration_min", "mean"),
                avg_hr=("avg_hr", "mean"),
                mean_max_hr=("max_hr", "mean"),
                load_total=("load", "sum"),
           )
           .sort_values("total_hours", ascending=False)
           .reset_index()
    )
    return overall, by_activity, sub

overall_tables = []
activity_tables = []

for name, start, end in windows:
    overall, by_act, _sub = summarize(name, start, end)
    if overall is not None:
        overall_tables.append(overall)
        by_act.insert(0, "block", name)
        activity_tables.append(by_act)

overall_out = pd.concat(overall_tables, ignore_index=True)
activity_out = pd.concat(activity_tables, ignore_index=True)

print("\n=== OVERALL COMPARISON ===")
print(overall_out.to_string(index=False))

print("\n=== BY ACTIVITY (top rows) ===")
print(activity_out.head(30).to_string(index=False))

overall_out.to_csv("out/build_comparison_overall.csv", index=False)
activity_out.to_csv("out/build_comparison_by_activity.csv", index=False)

print("\nSaved:")
print(" - out/build_comparison_overall.csv")
print(" - out/build_comparison_by_activity.csv")

