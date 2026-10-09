"""Small source-feasibility probe; not the project collector.

Run: python probe_api.py [--output NEW_DIRECTORY]
Requests 3 model menus, 5 option menus, 5 individual vehicles, and 1 XML menu.
The default writes into a new timestamped directory beside this script.
No ready-made dataset is downloaded. This is not a resumable bulk collector.
"""
import argparse
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "requests.jsonl"
BASE = "https://www.fueleconomy.gov/ws/rest/vehicle/"
DELAY = 1.0

def fetch(path, name, accept="application/json"):
    url = BASE + path
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "SE-2421-course-source-feasibility/0.1"})
    requested_at = datetime.now(timezone.utc).isoformat()
    with urllib.request.urlopen(req, timeout=40) as response:
        data = response.read()
        meta = {"url": url, "requested_at_utc": requested_at,
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "requested_accept": accept, "status": response.status,
                "response_headers": dict(response.headers), "file": name,
                "sha256": hashlib.sha256(data).hexdigest()}
    (ROOT / name).write_bytes(data)
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    time.sleep(DELAY)
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        parsed = data.decode("utf-8")
    print(f"Saved {name} ({len(data)} bytes)", flush=True)
    return parsed

def main():
    global ROOT, MANIFEST, DELAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, help="New destination directory; must not already exist")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests, >= 0.5")
    args = parser.parse_args()
    if args.delay < 0.5:
        parser.error("--delay must be at least 0.5 seconds")
    DELAY = args.delay
    ROOT = args.output or ROOT / ("recheck_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    ROOT.mkdir(parents=True, exist_ok=False)
    MANIFEST = ROOT / "requests.jsonl"
    for year, make in [(2015,"Honda"),(2025,"Toyota"),(2025,"Volvo")]:
        query = urllib.parse.urlencode({"year":year,"make":make})
        fetch("menu/model?" + query, f"models_{year}_{make}.json")
    for year, make, model in [(2015,"Honda","Fit"), (2025,"Toyota","RAV4"), (2025,"Toyota","Prius"), (2025,"Toyota","Prius PHEV"), (2025,"Volvo","XC90 B5 AWD")]:
        query = urllib.parse.urlencode({"year":year,"make":make,"model":model})
        name = f"options_{year}_{make}_{model.replace(' ', '_')}"
        payload = fetch("menu/options?" + query, name + ".json")
        items = payload.get("menuItem", [])
        items = [items] if isinstance(items, dict) else items
        if not isinstance(items, list) or not items:
            raise ValueError(f"Expected nonempty options menu: {name}")
        vehicle_id = str(items[0]["value"])
        if not vehicle_id.isdigit():
            raise ValueError(f"Non-numeric vehicle id: {vehicle_id}")
        record = fetch(vehicle_id, f"vehicle_{vehicle_id}.json")
        if str(record.get("id")) != vehicle_id or int(record.get("year")) != year:
            raise ValueError("Vehicle record disagrees with requested id/year")
        if record.get("make") != make or record.get("model") != model:
            raise ValueError("Vehicle record disagrees with requested make/model")
        if model == "Prius":
            fetch("menu/options?" + query, name + ".xml", "application/xml")
    print(f"Probe complete: 5 individual records. Output: {ROOT.resolve()}")

if __name__ == "__main__":
    main()
