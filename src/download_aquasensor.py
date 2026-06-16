import requests
import pandas as pd
import os
from io import StringIO
from datetime import datetime

# ── Credentials ───────────────────────────────────────────────
BASE_URL = "https://api.aquasensor.co.uk/aq.php"
USERNAME = "shu"
TOKEN    = "aebbf6305f9fce1d5591ee05a3448eff"

# ── Sensors (updated IDs from live API check 04/06/2026) ──────
SENSORS = {
    "sensor022": "Derwent_13",
    "941115":    "Derwent_13_50",
    "941205":    "Derwent_21",
}

# ── Date range ────────────────────────────────────────────────
FROM_DATE = "01-01-24"
TO_DATE   = datetime.today().strftime("%d-%m-%y")


def check_live_status():
    print("Checking live sensor status...")
    url = (f"{BASE_URL}?op=status"
           f"&username={USERNAME}&token={TOKEN}")
    r = requests.get(url, timeout=30)
    print(r.text)
    print()


def download_sensor(sensor_id, sensor_name):
    print(f"Downloading {sensor_name} ({sensor_id})...")
    url = (
        f"{BASE_URL}?op=readings"
        f"&username={USERNAME}"
        f"&token={TOKEN}"
        f"&sensorid={sensor_id}"
        f"&fromdate={FROM_DATE}"
        f"&todate={TO_DATE}"
    )
    r = requests.get(url, timeout=120)
    if r.status_code != 200 or len(r.text.strip()) < 20:
        print(f"  No data for {sensor_name}")
        return None
    df = pd.read_csv(StringIO(r.text))
    df['sensor_id']   = sensor_id
    df['sensor_name'] = sensor_name
    print(f"  Got {len(df)} rows")
    return df


def main():
    print("=" * 50)
    print("AquaSensor Data Download")
    print("=" * 50)

    # Check live status
    check_live_status()

    # Download each sensor
    all_dfs = []
    for sensor_id, sensor_name in SENSORS.items():
        df = download_sensor(sensor_id, sensor_name)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("ERROR: No data downloaded")
        return

    # Combine all sensors
    combined = pd.concat(all_dfs, ignore_index=True)

    # Clean column names
    combined.columns = combined.columns.str.strip().str.lower()
    combined = combined.rename(columns={
        'mg/l':    'dissolved_oxygen_mgl',
        'percent': 'dissolved_oxygen_pct',
    })

    # Create proper timestamp
    combined['timestamp'] = pd.to_datetime(
        combined['date'] + ' ' + combined['time'],
        dayfirst=True
    )

    # Sort
    combined = combined.sort_values(
        ['sensor_id', 'timestamp']
    ).reset_index(drop=True)

    # Save
    os.makedirs("data/raw", exist_ok=True)
    combined.to_csv(
        "data/raw/aquasensor_all_sensors.csv",
        index=False
    )

    # Save individual sensors
    for sid, sname in SENSORS.items():
        s = combined[combined['sensor_id'] == sid]
        if len(s) > 0:
            s.to_csv(f"data/raw/{sname}.csv", index=False)
            print(f"Saved {sname}: {len(s)} rows")

    print(f"\nDone!")
    print(f"Total rows: {len(combined)}")
    print(f"Date range: {combined['timestamp'].min()} "
          f"to {combined['timestamp'].max()}")
    print(f"Saved to: data/raw/aquasensor_all_sensors.csv")


main()