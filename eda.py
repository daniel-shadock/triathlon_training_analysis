from pathlib import Path
from lxml import etree

xml_path = Path("data/export.xml")

# Streaming parse so huge files don't blow up memory
workouts = 0
heart_rate = 0

for _, elem in etree.iterparse(str(xml_path), events=("end",), recover=True, huge_tree=True):
    if elem.tag == "Workout":
        workouts += 1
    elif elem.tag == "Record" and elem.attrib.get("type") == "HKQuantityTypeIdentifierHeartRate":
        heart_rate += 1

    # free memory
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

print("Workouts:", workouts)
print("Heart rate records:", heart_rate)

