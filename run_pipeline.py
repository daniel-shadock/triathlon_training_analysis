from __future__ import annotations

from pathlib import Path
from lxml import etree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------
# CONFIG
# ------------------
BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR / "data" / "export.xml"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

FAST_MODE = False  # set False when you need to reprocess XML

# Your build windows
BUILDS = {
    "2024 Build": ("2024-04-28", "2024-09-28"),
    "2025 Build": ("2025-02-13", "2025-07-13"),
    "2026 Build": ("2026-01-07", None),
}

# overlap threshold for clustering duplicates (0.5 is conservative; 0.4 is more aggressive)
OVERLAP_THRESHOLD = 0.5


# ------------------
# HELPERS: streaming parsers
# ------------------
def _iterparse(xml_path: Path, tag: str):
    for _, elem in etree.iterparse(
        str(xml_path),
        events=("end",),
        tag=tag,
        recover=True,
        huge_tree=True,
    ):
        yield elem
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


# ------------------
# EXPORT: Workouts + nested WorkoutStatistics (distance/energy)
# ------------------
def export_workouts_with_stats(xml_path: Path) -> pd.DataFrame:
    """
    Export <Workout> attributes plus nested <WorkoutStatistics> where available.
    Captures cycling distance if present as HKQuantityTypeIdentifierDistanceCycling.
    """
    rows = []
    for w in _iterparse(xml_path, "Workout"):
        row = dict(w.attrib)

        # Read nested stats
        stats: dict[str, tuple[float, str | None]] = {}
        for child in w:
            if child.tag == "WorkoutStatistics":
                t = child.attrib.get("type")
                s = child.attrib.get("sum")
                u = child.attrib.get("unit")
                if not t:
                    continue
                try:
                    stats[t] = (float(s), u)
                except Exception:
                    stats[t] = (np.nan, u)

        # Common stats we might care about
        cyc_dist, cyc_unit = stats.get("HKQuantityTypeIdentifierDistanceCycling", (np.nan, None))
        run_dist, run_unit = stats.get("HKQuantityTypeIdentifierDistanceWalkingRunning", (np.nan, None))
        energy, energy_unit = stats.get("HKQuantityTypeIdentifierActiveEnergyBurned", (np.nan, None))

        row["distance_cycling"] = cyc_dist
        row["distance_cycling_unit"] = cyc_unit
        row["distance_running"] = run_dist
        row["distance_running_unit"] = run_unit
        row["active_energy"] = energy
        row["active_energy_unit"] = energy_unit

        rows.append(row)

    df = pd.DataFrame(rows)

    # Parse datetimes
    for c in ["startDate", "endDate", "creationDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Duration
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # In your export, duration appears to already be minutes
    df["duration_min"] = df["duration"]

    # Friendly activity labels
    if "workoutActivityType" in df.columns:
        df["activity"] = df["workoutActivityType"].astype(str).str.replace("HKWorkoutActivityType", "", regex=False)

    return df


# ------------------
# EXPORT: Heart rate records
# ------------------
def export_heart_rate(xml_path: Path) -> pd.DataFrame:
    rows = []
    for r in _iterparse(xml_path, "Record"):
        if r.attrib.get("type") == "HKQuantityTypeIdentifierHeartRate":
            rows.append(dict(r.attrib))

    df = pd.DataFrame(rows)

    for c in ["startDate", "endDate", "creationDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ------------------
# ENRICH: attach HR to workouts
# ------------------
def attach_hr_to_workouts(workouts: pd.DataFrame, hr: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently assigns HR samples to workouts by time window via searchsorted,
    then computes per-workout avg/max HR + hr_samples.
    """
    w = workouts.copy().reset_index().rename(columns={"index": "workout_id"}).sort_values("startDate")
    r = hr.copy().sort_values("startDate")

    # Normalize tz off
    w["startDate"] = w["startDate"].dt.tz_localize(None)
    w["endDate"] = w["endDate"].dt.tz_localize(None)
    r["startDate"] = r["startDate"].dt.tz_localize(None)

    w_starts = w["startDate"].to_numpy()
    w_ends = w["endDate"].to_numpy()
    r_times = r["startDate"].to_numpy()

    idx = np.searchsorted(w_starts, r_times, side="right") - 1
    valid = idx >= 0

    workout_id = np.full(len(r), np.nan)
    candidate_pos = idx[valid]

    within = r_times[valid] <= w_ends[candidate_pos]
    workout_id_valid = w.iloc[candidate_pos[within]]["workout_id"].to_numpy()

    workout_id[np.where(valid)[0][within]] = workout_id_valid
    r["workout_id"] = workout_id

    hr_summary = (
        r.dropna(subset=["workout_id"])
        .groupby("workout_id")["value"]
        .agg(avg_hr="mean", max_hr="max", hr_samples="count")
        .reset_index()
    )

    enriched = w.merge(hr_summary, on="workout_id", how="left")
    return enriched


# ------------------
# MERGE CYCLING DUPLICATES:
# Keep Strava row, merge Apple Watch HR
# ------------------
def merge_cycling_duplicates_keep_strava_merge_hr(
    df: pd.DataFrame,
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Merge duplicate cycling workouts (Strava + Apple Watch) into ONE record:

    - Keep "most fields" from Strava workout (distance, etc.)
    - Merge HR fields (avg_hr, max_hr, hr_samples) from Apple Watch workout

    This must be run AFTER attach_hr_to_workouts() so both rows have HR summaries available.
    """
    out = df.copy()

    out["startDate"] = pd.to_datetime(out["startDate"], errors="coerce").dt.tz_localize(None)
    out["endDate"] = pd.to_datetime(out["endDate"], errors="coerce").dt.tz_localize(None)

    if "duration_min" not in out.columns:
        out["duration_min"] = pd.to_numeric(out.get("duration"), errors="coerce")

    for col in ["avg_hr", "max_hr", "hr_samples"]:
        if col not in out.columns:
            out[col] = np.nan

    is_cycling = out["workoutActivityType"].eq("HKWorkoutActivityTypeCycling")
    cycling = out[is_cycling].copy().sort_values("startDate").reset_index(drop=True)
    non_cycling = out[~is_cycling].copy()

    if cycling.empty:
        return out

    def text_blob(row) -> str:
        return " ".join([
            str(row.get("sourceName", "")),
            str(row.get("device", "")),
            str(row.get("sourceVersion", "")),
        ]).lower()

    def is_strava(row) -> bool:
        return "strava" in text_blob(row)

    def is_watch(row) -> bool:
        t = text_blob(row)
        return ("watch" in t) or ("apple" in t) or ("fitness" in t)

    def overlap_ratio(a, b) -> float:
        s1, e1 = a["startDate"], a["endDate"]
        s2, e2 = b["startDate"], b["endDate"]

        latest_start = max(s1, s2)
        earliest_end = min(e1, e2)
        overlap = (earliest_end - latest_start).total_seconds()
        if overlap <= 0:
            return 0.0

        dur1 = (e1 - s1).total_seconds()
        dur2 = (e2 - s2).total_seconds()
        shorter = min(dur1, dur2)
        if shorter <= 0:
            return 0.0

        return overlap / shorter

    # ---- build overlap clusters (connectivity/BFS) ----
    n = len(cycling)
    visited = np.zeros(n, dtype=bool)
    cluster_id = np.full(n, -1, dtype=int)
    cid = 0

    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        cluster_id[i] = cid

        while queue:
            a = queue.pop()
            for b in range(n):
                if visited[b]:
                    continue
                if overlap_ratio(cycling.iloc[a], cycling.iloc[b]) >= overlap_threshold:
                    visited[b] = True
                    cluster_id[b] = cid
                    queue.append(b)
        cid += 1

    cycling["cluster_id"] = cluster_id

    merged_rows = []

    for _, grp in cycling.groupby("cluster_id", sort=True):
        grp = grp.copy()

        # Choose primary (Strava if present; else longest duration)
        strava_rows = grp[grp.apply(is_strava, axis=1)]
        if not strava_rows.empty:
            primary = strava_rows.sort_values("duration_min", ascending=False).iloc[0].copy()
        else:
            primary = grp.sort_values("duration_min", ascending=False).iloc[0].copy()

        # Choose HR donor (Apple Watch if present; else row with most hr_samples)
        watch_rows = grp[grp.apply(is_watch, axis=1)]
        if not watch_rows.empty:
            hr_donor = watch_rows.sort_values("hr_samples", ascending=False).iloc[0].copy()
        else:
            hr_donor = grp.sort_values("hr_samples", ascending=False).iloc[0].copy()

        # Merge HR into primary if donor has values
        for col in ["avg_hr", "max_hr", "hr_samples"]:
            donor_val = hr_donor.get(col)
            if pd.notna(donor_val):
                primary[col] = donor_val

        # Expand time window to cover both (helps HR-windowing sanity)
        primary["startDate"] = grp["startDate"].min()
        primary["endDate"] = grp["endDate"].max()

        # Track HR provenance
        primary["hr_from"] = hr_donor.get("sourceName", "")

        merged_rows.append(primary)

    merged_cycling = pd.DataFrame(merged_rows)

    final = pd.concat([non_cycling, merged_cycling], ignore_index=True, sort=False)

    print(
        f"[merge] Cycling workouts: {len(cycling)} → {len(merged_cycling)} "
        f"(merged duplicates; overlap_threshold={overlap_threshold})"
    )

    return final


# ------------------
# LOAD + WEEKLY SUMMARY + OUTPUTS
# ------------------
def add_load(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an HR-based load proxy:
      intensity = scaled avg_hr between 10th and 90th percentile of your avg_hr distribution
      load = duration_min * intensity
    """
    df = enriched.copy()

    global_hr = df["avg_hr"].dropna()
    if len(global_hr) >= 50:
        hr_p10, hr_p90 = np.nanpercentile(global_hr, [10, 90])
    else:
        hr_p10, hr_p90 = (np.nan, np.nan)

    def hr_intensity(avg_hr):
        if np.isnan(avg_hr) or np.isnan(hr_p10) or np.isnan(hr_p90) or hr_p90 == hr_p10:
            return np.nan
        return float(np.clip((avg_hr - hr_p10) / (hr_p90 - hr_p10), 0, 1))

    df["intensity"] = df["avg_hr"].apply(hr_intensity)
    df["load"] = df["duration_min"] * df["intensity"]
    return df


def build_weekly_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces weekly time series per build aligned by week_number within build.
    """
    all_builds = []

    for build_name, (start, end) in BUILDS.items():
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
                load=("load", "sum"),
                avg_hr=("avg_hr", "mean"),
            )
            .sort_values("week")
        )

        weekly["week_number"] = np.arange(1, len(weekly) + 1)
        weekly["build"] = build_name
        all_builds.append(weekly)

    if not all_builds:
        return pd.DataFrame()

    return pd.concat(all_builds, ignore_index=True)


def save_one_comparison_table(weekly_df: pd.DataFrame) -> Path:
    """
    One wide week-by-week table, capped to shortest build length.
    """
    if weekly_df.empty:
        raise ValueError("weekly_df is empty; no data in the build windows.")

    N = int(weekly_df.groupby("build")["week_number"].max().min())
    weekly_cap = weekly_df[weekly_df["week_number"] <= N].copy()

    weekly_cap = weekly_cap.sort_values(["build", "week_number"])
    weekly_cap["cumulative_hours"] = weekly_cap.groupby("build")["hours"].cumsum()

    weekly_cap = weekly_cap[["week_number", "build", "hours", "load", "avg_hr", "cumulative_hours"]]

    wide = (
        weekly_cap.pivot_table(
            index="week_number",
            columns="build",
            values=["hours", "load", "avg_hr", "cumulative_hours"],
            aggfunc="first",
        )
    )

    # Make columns build-major: (Build, metric)
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

    out_path = OUT_DIR / "weekly_comparison_one_table_capped.csv"
    wide.to_csv(out_path)
    return out_path


def main():
    if FAST_MODE:
        print("FAST MODE: skipping XML parsing")

        enriched_path = OUT_DIR / "workouts_enriched.csv"
        weekly_path = OUT_DIR / "weekly_df.csv"

        if not enriched_path.exists():
            raise FileNotFoundError(f"Missing {enriched_path}. Run FULL MODE once to generate it.")
        if not weekly_path.exists():
            raise FileNotFoundError(f"Missing {weekly_path}. Run FULL MODE once to generate it.")

        print("1) Reading enriched workouts + weekly summaries...")
        enriched = pd.read_csv(enriched_path, parse_dates=["startDate", "endDate"])
        weekly_df = pd.read_csv(weekly_path, parse_dates=["week"])

        print("2) Saving dashboard plot...")
        plot_path = save_dashboard_plot(weekly_df, enriched)
        print(f"   saved: {plot_path}")

        print("\nDone.")
        return

    # -------- FULL MODE --------
    if not XML_PATH.exists():
        raise FileNotFoundError(f"Cannot find {XML_PATH}. Put export.xml at data/export.xml")

    print("FULL MODE: parsing export.xml")
    print(f"Reading: {XML_PATH}")

    print("1) Exporting workouts (with distance stats when present)...")
    workouts = export_workouts_with_stats(XML_PATH)
    workouts.to_csv(OUT_DIR / "workouts_raw.csv", index=False)
    print(f"   workouts raw: {len(workouts):,}")

    print("2) Exporting heart rate records...")
    hr = export_heart_rate(XML_PATH)
    hr.to_csv(OUT_DIR / "heart_rate.csv", index=False)
    print(f"   HR samples: {len(hr):,}")

    print("3) Attaching HR to workouts...")
    enriched = attach_hr_to_workouts(workouts, hr)

    # normalize tz for downstream
    enriched["startDate"] = enriched["startDate"].dt.tz_localize(None)
    enriched["endDate"] = enriched["endDate"].dt.tz_localize(None)

    print("4) Merging cycling duplicates (keep Strava + merge Watch HR)...")
    enriched = merge_cycling_duplicates_keep_strava_merge_hr(
        enriched,
        overlap_threshold=OVERLAP_THRESHOLD,
    )

    print("5) Computing load...")
    enriched = add_load(enriched)

    enriched.to_csv(OUT_DIR / "workouts_enriched.csv", index=False)
    print(f"   saved: {OUT_DIR / 'workouts_enriched.csv'}")

    print("6) Building weekly summaries...")
    weekly_df = build_weekly_df(enriched)
    weekly_df.to_csv(OUT_DIR / "weekly_df.csv", index=False)
    print(f"   saved: {OUT_DIR / 'weekly_df.csv'}")

    print("7) Saving capped comparison table...")
    table_path = save_one_comparison_table(weekly_df)
    print(f"   saved: {table_path}")

    print("8) Saving dashboard plot...")
    plot_path = save_dashboard_plot(weekly_df, enriched)
    print(f"   saved: {plot_path}")

    print("\nDone.")

def save_dashboard_plot(weekly_df: pd.DataFrame, enriched: pd.DataFrame) -> Path:
    """
    Saves a 2x4 dashboard figure as PNG with Economist-like styling.
    Includes:
      (1) Weekly Hours
      (2) Weekly HR-Based Load
      (3) Avg HR
      (4) Cumulative Hours
      (5) Total Hours by Event (STACKED, capped)
      (6) Event Composition (100% STACKED)
      (7) Notes
      (8) blank
    All plots are capped to the shortest build length (weeks 1..N).
    """
    if weekly_df.empty:
        raise ValueError("weekly_df is empty; cannot plot.")
    if enriched.empty:
        raise ValueError("enriched is empty; cannot plot event totals.")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # ------------------
    # Economist-ish styling (muted, clean, editorial)
    # ------------------
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "DejaVu Sans",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.85,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })

    # Muted palette for event stacks (explicit to avoid garish defaults)
    EVENT_COLORS = {
        "Swimming":  "#1f77b4",  # muted blue
        "Cycling":   "#3ebcd2",  # muted cyan
        "Running":   "#2ca02c",  # muted green
        "Strength":  "#7f7f7f",  # neutral gray
    }

    # ------------------
    # Cap to shortest build length (weeks 1..N)
    # ------------------
    N = int(weekly_df.groupby("build")["week_number"].max().min())
    plot_df = weekly_df[weekly_df["week_number"] <= N].copy()
    plot_df = plot_df.sort_values(["build", "week_number"])
    plot_df["cumulative_hours"] = plot_df.groupby("build")["hours"].cumsum()

    week_map = plot_df[["build", "week", "week_number"]].copy()

    # ------------------
    # Prepare workout-level data for event totals (capped weeks only)
    # ------------------
    df = enriched.copy()
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce").dt.tz_localize(None)

    if "activity" not in df.columns and "workoutActivityType" in df.columns:
        df["activity"] = (
            df["workoutActivityType"]
            .astype(str)
            .str.replace("HKWorkoutActivityType", "", regex=False)
        )

    df["duration_min"] = pd.to_numeric(df.get("duration_min", df.get("duration")), errors="coerce")
    df["hours"] = df["duration_min"] / 60.0

    def assign_build(ts: pd.Timestamp) -> str | None:
        for b, (start, end) in BUILDS.items():
            if ts >= pd.Timestamp(start) and (end is None or ts <= pd.Timestamp(end)):
                return b
        return None

    df["build"] = df["startDate"].apply(assign_build)
    df = df.dropna(subset=["build"]).copy()
    df["week"] = df["startDate"].dt.to_period("W").dt.start_time
    df = df.merge(week_map, on=["build", "week"], how="inner")

    def bucket_event(a: str) -> str:
        a = str(a).strip().lower()
        if "cycling" in a:
            return "Cycling"
        if "running" in a:
            return "Running"
        if "swimming" in a or "water" in a:
            return "Swimming"
        return "Strength"  # strength + walking + other

    df["event_bucket"] = df["activity"].apply(bucket_event)
    event_totals = df.groupby(["build", "event_bucket"], as_index=False)["hours"].sum()

    build_order = list(BUILDS.keys())
    event_order = ["Swimming", "Cycling", "Running", "Strength"]

    ev_wide = (
        event_totals.pivot_table(index="build", columns="event_bucket", values="hours", aggfunc="sum")
        .reindex(build_order)
        .reindex(columns=event_order, fill_value=0.0)
        .fillna(0.0)
    )

    percent = ev_wide.copy()
    row_sums = percent.sum(axis=1).replace(0, np.nan)
    percent = (percent.div(row_sums, axis=0) * 100.0).fillna(0.0)

    # ------------------
    # Plotting layout
    # ------------------
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

    def economist_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="#333333")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

    def label_line_end(ax, x, y, label, pad=0.35):
        """
        Put a text label at the last point of a line.
        pad controls small right shift.
        """
        if len(x) == 0 or len(y) == 0:
            return
        ax.text(
            x[-1] + pad,
            y[-1],
            str(label),
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
            clip_on=False,
        )

    # 1) Weekly Hours (direct labels, no legend)
    max_x = 0
    for b in build_order:
        s = plot_df[plot_df["build"] == b]
        if s.empty:
            continue
        x = s["week_number"].to_numpy()
        y = s["hours"].to_numpy()
        ax1.plot(x, y)
        label_line_end(ax1, x, y, b)
        max_x = max(max_x, int(x.max()))
    ax1.set_title("Weekly Training Hours (capped)")
    ax1.set_xlabel("Week #")
    ax1.set_ylabel("Hours")
    ax1.set_xlim(1, max_x + 2)  # room for labels
    economist_axes(ax1)

    # 2) Weekly Load
    max_x = 0
    for b in build_order:
        s = plot_df[plot_df["build"] == b]
        if s.empty:
            continue
        x = s["week_number"].to_numpy()
        y = s["load"].to_numpy()
        ax2.plot(x, y)
        label_line_end(ax2, x, y, b)
        max_x = max(max_x, int(x.max()))
    ax2.set_title("Weekly HR-Based Load (capped)")
    ax2.set_xlabel("Week #")
    ax2.set_ylabel("Load")
    ax2.set_xlim(1, max_x + 2)
    economist_axes(ax2)

    # 3) Avg HR
    max_x = 0
    for b in build_order:
        s = plot_df[plot_df["build"] == b]
        if s.empty:
            continue
        x = s["week_number"].to_numpy()
        y = s["avg_hr"].to_numpy()
        ax3.plot(x, y)
        label_line_end(ax3, x, y, b)
        max_x = max(max_x, int(x.max()))
    ax3.set_title("Average Workout HR (capped)")
    ax3.set_xlabel("Week #")
    ax3.set_ylabel("Avg HR (bpm)")
    ax3.set_xlim(1, max_x + 2)
    economist_axes(ax3)

    # 4) Cumulative Hours
    max_x = 0
    for b in build_order:
        s = plot_df[plot_df["build"] == b]
        if s.empty:
            continue
        x = s["week_number"].to_numpy()
        y = s["cumulative_hours"].to_numpy()
        ax4.plot(x, y)
        label_line_end(ax4, x, y, b)
        max_x = max(max_x, int(x.max()))
    ax4.set_title("Cumulative Training Hours (capped)")
    ax4.set_xlabel("Week #")
    ax4.set_ylabel("Cumulative Hours")
    ax4.set_xlim(1, max_x + 2)
    economist_axes(ax4)

    # 5) Stacked hours by event (absolute)
    x = np.arange(len(ev_wide.index))
    width = 0.55
    bottom = np.zeros(len(ev_wide.index))

    for ev in event_order:
        vals = ev_wide[ev].to_numpy()
        ax5.bar(x, vals, width=width, bottom=bottom, color=EVENT_COLORS[ev], edgecolor="none")
        bottom += vals

    ax5.set_title("Total Hours by Event (capped)")
    ax5.set_xlabel("Build")
    ax5.set_ylabel("Hours (weeks 1..N)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(ev_wide.index)
    economist_axes(ax5)

    ax5.legend(
        event_order,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Event",
    )

    # 6) 100% stacked composition
    x = np.arange(len(percent.index))
    bottom = np.zeros(len(percent.index))

    for ev in event_order:
        vals = percent[ev].to_numpy()
        ax6.bar(x, vals, width=width, bottom=bottom, color=EVENT_COLORS[ev], edgecolor="none")
        bottom += vals

    ax6.set_title("Event Composition (% of total hours)")
    ax6.set_xlabel("Build")
    ax6.set_ylabel("Percent")
    ax6.set_xticks(x)
    ax6.set_xticklabels(percent.index)
    ax6.set_ylim(0, 100)
    economist_axes(ax6)

    ax6.legend(
        event_order,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Event",
    )

    # 7) Notes
    overlap = globals().get("OVERLAP_THRESHOLD", None)
    note_lines = [f"Capped to N = {N} weeks"]
    if overlap is not None:
        note_lines.append(f"Overlap threshold = {overlap}")
    ax7.axis("off")
    ax7.text(0.0, 0.9, "\n".join(note_lines), fontsize=10, color="#333333", transform=ax7.transAxes)

    # 8) blank
    ax8.axis("off")

    # Extra room on right for legends; extra room on left/right for line end labels
    plt.tight_layout(rect=[0, 0, 0.86, 1])

    out_path = OUT_DIR / "build_dashboard_2x4.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    main()