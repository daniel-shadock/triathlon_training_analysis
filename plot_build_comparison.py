import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")

# ---------- Load enriched workouts ----------
df = pd.read_csv(
    "out/workouts_enriched.csv",
    parse_dates=["startDate", "endDate"],
    low_memory=False
)

# normalize timezone for weekly bucketing
df["startDate"] = df["startDate"].dt.tz_localize(None)

# Ensure duration_min exists (some exports only have duration)
if "duration_min" not in df.columns:
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    # Your export's duration appears to be minutes already
    df["duration_min"] = df["duration"]

# ---------- Create HR-based load (same idea as earlier) ----------
global_hr = df["avg_hr"].dropna()
if len(global_hr) >= 50:
    hr_p10, hr_p90 = np.nanpercentile(global_hr, [10, 90])
else:
    hr_p10, hr_p90 = (np.nan, np.nan)

def hr_intensity(avg_hr):
    if np.isnan(avg_hr) or np.isnan(hr_p10) or np.isnan(hr_p90) or hr_p90 == hr_p10:
        return np.nan
    return np.clip((avg_hr - hr_p10) / (hr_p90 - hr_p10), 0, 1)

df["intensity"] = df["avg_hr"].apply(hr_intensity)
df["load"] = df["duration_min"] * df["intensity"]

# ---------- Define build windows ----------
builds = {
    "2024 Build": ("2024-04-28", "2024-09-28"),
    "2025 Build": ("2025-02-13", "2025-07-13"),
    "2026 Build": ("2026-01-07", None),
}

all_builds = []

for name, (start, end) in builds.items():
    sub = df[df["startDate"] >= pd.Timestamp(start)].copy()
    if end is not None:
        sub = sub[sub["startDate"] <= pd.Timestamp(end)]

    if sub.empty:
        continue

    sub = sub.sort_values("startDate")
    sub["week"] = sub["startDate"].dt.to_period("W").dt.start_time

    weekly = (
        sub.groupby("week", as_index=False)
           .agg(
                hours=("duration_min", lambda x: x.sum() / 60),
                avg_hr=("avg_hr", "mean"),
                load=("load", "sum")
           )
           .sort_values("week")
    )

    weekly["week_number"] = np.arange(1, len(weekly) + 1)
    weekly["build"] = name
    all_builds.append(weekly)


weekly_df = pd.concat(all_builds, ignore_index=True)

# =========================
# ONE WEEK-BY-WEEK COMPARISON TABLE (capped to current weeks)
# =========================

# Cap to shortest build length (your current number of weeks)
N = int(weekly_df.groupby("build")["week_number"].max().min())
weekly_cap = weekly_df[weekly_df["week_number"] <= N].copy()

# Add cumulative hours within each build (using capped data)
weekly_cap = weekly_cap.sort_values(["build", "week_number"])
weekly_cap["cumulative_hours"] = weekly_cap.groupby("build")["hours"].cumsum()

# Keep only the metrics you want in the table
weekly_cap = weekly_cap[["week_number", "build", "hours", "load", "avg_hr", "cumulative_hours"]]

# Pivot to ONE wide table with MultiIndex columns (build, metric)
wide = (
    weekly_cap
    .pivot_table(index="week_number", columns="build",
                 values=["hours", "load", "avg_hr", "cumulative_hours"],
                 aggfunc="first")
    .sort_index(axis=1, level=1)  # sort builds within each metric
)

# Reorder columns to be Build-major (each build has all metrics)
wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

# Optional: nicer rounding
wide = wide.round({
    ("2024 Build", "hours"): 2,
    ("2024 Build", "load"): 1,
    ("2024 Build", "avg_hr"): 1,
    ("2024 Build", "cumulative_hours"): 2,
    ("2025 Build", "hours"): 2,
    ("2025 Build", "load"): 1,
    ("2025 Build", "avg_hr"): 1,
    ("2025 Build", "cumulative_hours"): 2,
    ("2026 Build", "hours"): 2,
    ("2026 Build", "load"): 1,
    ("2026 Build", "avg_hr"): 1,
    ("2026 Build", "cumulative_hours"): 2,
})

print(f"\nCapped all builds to N = {N} weeks.\n")
print(wide.to_string())

# Save as CSV (Excel-friendly: MultiIndex columns become two header rows)
wide.to_csv("out/weekly_comparison_one_table_capped.csv")
print("\nSaved: out/weekly_comparison_one_table_capped.csv")

# =========================
# PLOTTING: 4 graphs on one screen (2x2)
# =========================

# Add cumulative hours per build
weekly_df = weekly_df.sort_values(["build", "week_number"]).copy()
weekly_df["cumulative_hours"] = weekly_df.groupby("build")["hours"].cumsum()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

build_order = list(builds.keys())  # keeps 2024, 2025, 2026 order

# 1) Weekly Hours
for b in build_order:
    s = weekly_df[weekly_df["build"] == b]
    ax1.plot(s["week_number"], s["hours"], label=b)
ax1.set_title("Weekly Training Hours")
ax1.set_xlabel("Week Number Within Build")
ax1.set_ylabel("Hours")
ax1.grid(True)

# 2) Weekly Load
for b in build_order:
    s = weekly_df[weekly_df["build"] == b]
    ax2.plot(s["week_number"], s["load"], label=b)
ax2.set_title("Weekly HR-Based Load")
ax2.set_xlabel("Week Number Within Build")
ax2.set_ylabel("Load (min × intensity)")
ax2.grid(True)

# 3) Avg HR
for b in build_order:
    s = weekly_df[weekly_df["build"] == b]
    ax3.plot(s["week_number"], s["avg_hr"], label=b)
ax3.set_title("Average Workout HR")
ax3.set_xlabel("Week Number Within Build")
ax3.set_ylabel("Avg HR (bpm)")
ax3.grid(True)

# 4) Cumulative Hours
for b in build_order:
    s = weekly_df[weekly_df["build"] == b]
    ax4.plot(s["week_number"], s["cumulative_hours"], label=b)
ax4.set_title("Cumulative Training Hours")
ax4.set_xlabel("Week Number Within Build")
ax4.set_ylabel("Cumulative Hours")
ax4.grid(True)

# Shared legend once (top center)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

