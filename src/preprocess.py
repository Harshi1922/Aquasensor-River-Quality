import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer

print("=" * 55)
print("AquaSensor Feature Engineering Pipeline")
print("=" * 55)


def load_data():
    """Load AquaSensor sensor data."""
    df = pd.read_csv(
        "data/raw/aquasensor_all_sensors.csv",
        parse_dates=['timestamp'],
        low_memory=False
    )
    df = df.sort_values(
        ['sensor_id', 'timestamp']
    ).reset_index(drop=True)
    df = df.rename(columns={'temperature': 'water_temp_c'})
    print(f"Loaded: {len(df)} rows, "
          f"{df['sensor_id'].nunique()} sensors")
    print(f"Raw columns: {list(df.columns)}")
    return df


def add_target(df):
    """Add target: DO 15 minutes ahead."""
    df['do_next_15min'] = df.groupby('sensor_id')[
        'dissolved_oxygen_mgl'
    ].shift(-1)
    print("Target added: do_next_15min")
    return df


def add_time_features(df):
    """
    Extract time features.
    month, season, day_of_year are temporary —
    consumed by add_season_proxy, then dropped.
    """
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # Temporary columns for season_proxy
    df['month']       = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['season']      = df['month'].map({
        12:0, 1:0, 2:0,
        3:1,  4:1, 5:1,
        6:2,  7:2, 8:2,
        9:3, 10:3, 11:3
    })
    print("Time features added: hour, day_of_week, is_weekend "
          "(month/season/day_of_year held for season_proxy)")
    return df


def add_weather_features(df):
    """Add weather features from saved CSV."""
    try:
        weather = pd.read_csv(
            "data/raw/weather_data.csv",
            parse_dates=['timestamp']
        )
        weather['ts_hour'] = weather['timestamp'].dt.floor('h')
        df['ts_hour']      = df['timestamp'].dt.floor('h')

        df = df.merge(
            weather[['ts_hour', 'air_temp_c', 'rainfall_mm',
                     'wind_speed_kmh', 'sunshine_wm2',
                     'cloud_cover_pct']],
            on='ts_hour', how='left'
        )
        df = df.drop(columns=['ts_hour'])
        print("Weather features added: air_temp_c, sunshine_wm2, "
              "rainfall_mm, wind_speed_kmh, cloud_cover_pct")
    except Exception as e:
        print(f"Weather merge failed: {e}")
        for col in ['air_temp_c', 'sunshine_wm2', 'rainfall_mm',
                    'wind_speed_kmh', 'cloud_cover_pct']:
            df[col] = np.nan
    return df


def add_river_features(df):
    """Add river level from EA data."""
    try:
        river = pd.read_csv(
            "data/raw/govt_river_derwent.csv",
            parse_dates=['timestamp']
        )
        # Strip timezone if present — fixes UTC vs naive merge error
        river['timestamp'] = river['timestamp'].dt.tz_localize(None)

        river['ts_15'] = river['timestamp'].dt.floor('15min')
        df['ts_15']    = df['timestamp'].dt.floor('15min')

        river_cols = ['ts_15']
        if 'river_level_m' in river.columns:
            river_cols.append('river_level_m')
        if 'river_flow_m3s' in river.columns:
            river_cols.append('river_flow_m3s')

        df = df.merge(river[river_cols], on='ts_15', how='left')
        df = df.drop(columns=['ts_15'])

        if 'river_level_m' not in df.columns:
            df['river_level_m'] = np.nan
        if 'river_flow_m3s' not in df.columns:
            df['river_flow_m3s'] = np.nan

        print(f"River features added: "
              f"{[c for c in river_cols if c != 'ts_15']}")
    except Exception as e:
        print(f"River merge failed: {e}")
        df['river_level_m']  = np.nan
        df['river_flow_m3s'] = np.nan
    return df


def add_season_proxy(df):
    """
    Single season proxy combining month, season, cloud_cover_pct,
    and day_of_year — as specified by lecturer 05/06/2026.
    Normalised weighted average, range 0-1.
    """
    month_norm  = (df['month'] - 1) / 11
    season_norm = df['season'] / 3
    cloud_norm  = df['cloud_cover_pct'].fillna(50) / 100
    day_norm    = (df['day_of_year'] - 1) / 364

    df['season_proxy'] = (
        0.30 * month_norm  +
        0.30 * season_norm +
        0.20 * day_norm    +
        0.20 * cloud_norm
    )
    print("Season proxy added: season_proxy "
          "(month 30%, season 30%, day_of_year 20%, "
          "cloud_cover_pct 20%)")
    return df


def add_sensor_name(df):
    """
    Add human-readable sensor_name if not already in data.
    Update this map to match your actual sensor IDs.
    """
    if 'sensor_name' not in df.columns:
        sensor_name_map = {
            'S1': 'Derwent_Upstream',
            'S2': 'Derwent_Midpoint',
            'S3': 'Derwent_Downstream',
            'S4': 'Derwent_Tributary',
        }
        df['sensor_name'] = df['sensor_id'].map(
            sensor_name_map
        ).fillna(df['sensor_id'].astype(str))
        print("Sensor names added from map")
    else:
        print("Sensor names already present in raw data")
    return df


def add_pollution_label(df, threshold=4.0):
    """Add binary pollution alert label."""
    df['pollution_alert'] = (
        df['dissolved_oxygen_mgl'] < threshold
    ).astype(int)
    n   = df['pollution_alert'].sum()
    pct = df['pollution_alert'].mean()
    print(f"Pollution alerts: {n} ({pct:.1%} of readings)")
    return df


def add_anomaly_type(df, do_low=4.0, do_high=12.0, temp_high=25.0):
    """
    Categorical anomaly label.
    Categories: low_DO, high_DO, high_temp, multi, normal
    """
    temp_col = None
    for candidate in ['water_temp_c', 'water_temperature_c',
                      'temperature_c', 'temp_c']:
        if candidate in df.columns:
            temp_col = candidate
            break

    low_do    = df['dissolved_oxygen_mgl'] < do_low
    high_do   = df['dissolved_oxygen_mgl'] > do_high
    high_temp = pd.Series(False, index=df.index)
    if temp_col:
        high_temp = df[temp_col] > temp_high

    conditions = [low_do & high_temp, low_do, high_do, high_temp]
    choices    = ['multi', 'low_DO', 'high_DO', 'high_temp']
    df['anomaly_type'] = np.select(
        conditions, choices, default='normal'
    )
    print(f"Anomaly types:\n"
          f"{df['anomaly_type'].value_counts().to_string()}")
    return df


def impute(df):
    """KNN imputation on the 10 required columns only."""
    required_numeric = [
    'water_temp_c',
    'air_temp_c',
    'sunshine_wm2',
    'hour',
    'dissolved_oxygen_mgl',
    'season_proxy',
]

    # Only impute columns that exist and have some data
    cols = [c for c in required_numeric
            if c in df.columns and df[c].notna().any()]

    print(f"Imputing {len(cols)} columns: {cols}")

    imputer  = KNNImputer(n_neighbors=5)
    imputed  = imputer.fit_transform(df[cols])
    df[cols] = imputed

    print(f"Imputation done. Nulls remaining: "
          f"{df[cols].isnull().sum().sum()}")
    return df


def main():
    df = load_data()
    df = add_target(df)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_river_features(df)
    df = add_season_proxy(df)

    # Drop temporary columns used only for season_proxy
    df = df.drop(
        columns=['month', 'season', 'day_of_year'],
        errors='ignore'
    )

    df = add_sensor_name(df)
    df = add_pollution_label(df)
    df = add_anomaly_type(df)
    df = impute(df)

    # Remove rows with no target
    df = df.dropna(subset=['do_next_15min'])

    # Keep only required columns
    final_cols = [
        'sensor_id',
        'sensor_name',
        'timestamp',
        'water_temp_c',
        'air_temp_c',
        'sunshine_wm2',
        'hour',
        'dissolved_oxygen_mgl',
        'pollution_alert',
        'anomaly_type',
        'season_proxy',
        'do_next_15min',
    ]

    df = df[final_cols]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/aquasensor_final.csv", index=False)

    print(f"\n{'=' * 55}")
    print("PIPELINE COMPLETE")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nAll columns:")
    for col in df.columns:
        print(f"  {col}")
    print(f"\nSaved: data/processed/aquasensor_final.csv")


main()