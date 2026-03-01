import pandas as pd

df = pd.read_csv("out/workouts_enriched.csv", parse_dates=["startDate"])

print(df[["workoutActivityType", "duration_min", "avg_hr", "max_hr"]].tail())

df["week"] = df["startDate"].dt.tz_localize(None).dt.to_period("W").dt.start_time

weekly_hr = df.groupby("week")["avg_hr"].mean()

print(weekly_hr.tail(12))

