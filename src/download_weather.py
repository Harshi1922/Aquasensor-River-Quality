import requests
import pandas as pd
import os
from datetime import datetime

FROM = "2024-01-01"
TO   = datetime.today().strftime("%Y-%m-%d")


def main():
    print("=" * 50)
    print("Open-Meteo Weather Data Download")
    print("Location: River Derwent, Hathersage")
    print("=" * 50)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   53.33,
        "longitude":  -1.65,
        "start_date": FROM,
        "end_date":   TO,
        "hourly": [
            "temperature_2m",      # air temperature
            "precipitation",       # rainfall
            "windspeed_10m",       # wind speed
            "shortwave_radiation", # sunshine (lecturer confirmed)
            "cloudcover"           # cloud cover
        ]
    }

    print("Downloading weather data...")
    try:
        r = requests.get(url, params=params, timeout=60)
        data = r.json()

        df = pd.DataFrame({
            'timestamp':       pd.to_datetime(
                                   data['hourly']['time']),
            'air_temp_c':      data['hourly']['temperature_2m'],
            'rainfall_mm':     data['hourly']['precipitation'],
            'wind_speed_kmh':  data['hourly']['windspeed_10m'],
            'sunshine_wm2':    data['hourly'][
                                   'shortwave_radiation'],
            'cloud_cover_pct': data['hourly']['cloudcover'],
        })

        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/weather_data.csv", index=False)

        print(f"Saved: data/raw/weather_data.csv")
        print(f"Rows: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} "
              f"to {df['timestamp'].max()}")
        print(df.head())

    except Exception as e:
        print(f"ERROR: {e}")


main()