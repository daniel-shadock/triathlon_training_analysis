from pathlib import Path
from lxml import etree
from collections import Counter

xml_path = Path("data") / "export.xml"

types = Counter()
n = 0

for _, elem in etree.iterparse(str(xml_path), events=("end",), tag="Record", recover=True, huge_tree=True):
    t = elem.attrib.get("type")
    if t:
        types[t] += 1
    n += 1

    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

    if n >= 200_000:
        break

print("Unique Record types sampled:", len(types))
for t, c in types.most_common(40):
    print(f"{c:>8}  {t}")