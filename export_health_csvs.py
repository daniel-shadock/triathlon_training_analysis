from pathlib import Path
from lxml import etree
import pandas as pd


XML_PATH = Path("data") / "export.xml"
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


def stream_workouts(xml_path: Path):
    for _, elem in etree.iterparse(str(xml_path), events=("end",), tag="Workout", recover=True, huge_tree=True):
        yield dict(elem.attrib)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


def stream_hr(xml_path: Path):
    # Heart rate records: HKQuantityTypeIdentifierHeartRate
    for _, elem in etree.iterparse(str(xml_path), events=("end",), tag="Record", recover=True, huge_tree=True):
        if elem.attrib.get("type") == "HKQuantityTypeIdentifierHeartRate":
            yield dict(elem.attrib)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


def main():
    print("Reading workouts...")
    workouts = pd.DataFrame(stream_workouts(XML_PATH))
    for c in ["startDate", "endDate", "creationDate"]:
        if c in workouts.columns:
            workouts[c] = pd.to_datetime(workouts[c], errors="coerce")
    if "duration" in workouts.columns:
        workouts["duration"] = pd.to_numeric(workouts["duration"], errors="coerce")
        workouts["duration_min"] = workouts["duration"] / 60

    workouts_out = OUT_DIR / "workouts.csv"
    workouts.to_csv(workouts_out, index=False)
    print(f"Saved {len(workouts):,} workouts -> {workouts_out}")

    print("Reading heart rate records...")
    hr = pd.DataFrame(stream_hr(XML_PATH))
    for c in ["startDate", "endDate", "creationDate"]:
        if c in hr.columns:
            hr[c] = pd.to_datetime(hr[c], errors="coerce")
    if "value" in hr.columns:
        hr["value"] = pd.to_numeric(hr["value"], errors="coerce")

    hr_out = OUT_DIR / "heart_rate.csv"
    hr.to_csv(hr_out, index=False)
    print(f"Saved {len(hr):,} HR records -> {hr_out}")

    print("Done.")


if __name__ == "__main__":
    main()