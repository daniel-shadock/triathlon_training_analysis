from pathlib import Path
from lxml import etree
import pandas as pd
import numpy as np

XML_PATH = Path("data") / "export.xml"

print("Loading workouts...")
workouts = []

for _, elem in etree.iterparse(str(XML_PATH), events=("end",), tag="Workout", recover=True, huge_tree=True):
    workouts.append(dict(elem.attrib))
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

workouts = pd.DataFrame(workouts)

# clean
workouts["startDate"] = pd.to_datetime(workouts["startDate"])
workouts["endDate"] = pd.to_datetime(workouts["endDate"])
workouts["duration"] = pd.to_numeric(workouts["duration"], errors="coerce")

# assume durationUnit is minutes (based on your data)
workouts["duration_min"] = workouts["duration"]

workouts = workouts.reset_index().rename(columns={"index": "workout_id"})

print("Loading heart rate records...")
hr_records = []

for _, elem in etree.iterparse(str(XML_PATH), events=("end",), tag="Record", recover=True, huge_tree=True):
    if elem.attrib.get("type") == "HKQuantityTypeIdentifierHeartRate":
        hr_records.append(dict(elem.attrib))
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

hr = pd.DataFrame(hr_records)

hr["startDate"] = pd.to_datetime(hr["startDate"])
hr["value"] = pd.to_numeric(hr["value"])

print("Assigning HR to workouts...")

hr["workout_id"] = np.nan

for i, row in workouts.iterrows():
    mask = (hr["startDate"] >= row["startDate"]) & (hr["startDate"] <= row["endDate"])
    hr.loc[mask, "workout_id"] = row["workout_id"]

print("Computing HR summary per workout...")

hr_summary = (
    hr.dropna(subset=["workout_id"])
      .groupby("workout_id")["value"]
      .agg(avg_hr="mean", max_hr="max", std_hr="std")
      .reset_index()
)

final = workouts.merge(hr_summary, on="workout_id", how="left")

final.to_csv("out/workouts_enriched.csv", index=False)

print("Saved enriched workouts to out/workouts_enriched.csv")

