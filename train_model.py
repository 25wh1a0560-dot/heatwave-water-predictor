import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "hyderabad_2022_water_weather_training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "water_demand_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("HYDERABAD HEATWAVE DEMAND MODEL")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Original rows:", len(df))


# ============================================================
# CLEAN
# ============================================================

df["section"] = (
    df["section"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

numeric_cols = [
    "water_demand",
    "avg_temperature_c",
    "max_temperature_c",
    "avg_humidity_pct",
    "total_rain_mm",
    "month"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=numeric_cols + ["section", "date"]
).copy()

# Remove invalid negative demand
df = df[
    df["water_demand"] >= 0
].copy()

df = df.sort_values(
    ["section", "date"]
).reset_index(drop=True)


print("Clean rows:", len(df))


# ============================================================
# SECTION STATISTICS
# ============================================================

section_stats = {}

for section, group in df.groupby("section"):

    demand = group["water_demand"]

    section_stats[section] = {

        "median_demand":
            float(demand.median()),

        "mean_demand":
            float(demand.mean()),

        "p25":
            float(demand.quantile(0.25)),

        "p75":
            float(demand.quantile(0.75))
    }


# ============================================================
# MONTHLY STATISTICS
# ============================================================

monthly_stats = {}

for month, group in df.groupby("month"):

    monthly_stats[int(month)] = {

        "median_demand":
            float(group["water_demand"].median()),

        "mean_demand":
            float(group["water_demand"].mean())
    }


# ============================================================
# TEMPERATURE EFFECT
# ============================================================
#
# We estimate a simple overall heat sensitivity from the
# historical data.
#
# We compare demand during warmer observations with the
# overall median demand.
#
# This is deliberately conservative.
# ============================================================

overall_median = float(
    df["water_demand"].median()
)

# Median maximum temperature
normal_max_temp = float(
    df["max_temperature_c"].median()
)

print(
    "Normal maximum temperature:",
    round(normal_max_temp, 2)
)

print(
    "Overall median demand:",
    round(overall_median, 2)
)


# ============================================================
# HEAT SENSITIVITY
# ============================================================

# Create temperature bins

df["temp_excess"] = (
    df["max_temperature_c"]
    - normal_max_temp
)

# Only consider hotter-than-normal observations

hot_data = df[
    df["temp_excess"] > 0
].copy()


if len(hot_data) > 10:

    # Demand ratio relative to overall median

    hot_ratio = (
        hot_data["water_demand"]
        / overall_median
    )

    # Remove extreme outliers

    hot_ratio = hot_ratio.clip(
        0.5,
        1.5
    )

    # Average response per degree

    heat_sensitivity = (
        (hot_ratio - 1)
        /
        hot_data["temp_excess"].replace(
            0,
            np.nan
        )
    ).median()

else:

    heat_sensitivity = 0.02


# Conservative bounds

heat_sensitivity = float(
    np.clip(
        heat_sensitivity,
        0.005,
        0.03
    )
)


print(
    "Heat sensitivity:",
    round(heat_sensitivity, 4)
)


# ============================================================
# AREA HEAT SENSITIVITY
# ============================================================

area_sensitivity = {}


for section, group in df.groupby("section"):

    if len(group) >= 3:

        normal = group[
            "water_demand"
        ].median()

        hot = group[
            group["max_temperature_c"]
            >= normal_max_temp
        ]["water_demand"]

        if len(hot) >= 2 and normal > 0:

            ratio = (
                hot.median()
                / normal
            )

            # Convert ratio to sensitivity

            sensitivity = (
                ratio - 1
            )

            # Conservative area-specific range

            sensitivity = float(
                np.clip(
                    sensitivity,
                    -0.10,
                    0.30
                )
            )

        else:

            sensitivity = 0.0

    else:

        sensitivity = 0.0


    area_sensitivity[section] = sensitivity


# ============================================================
# SAVE MODEL
# ============================================================

model_data = {

    "model_type":
        "hybrid_heatwave_scenario",

    "sections":
        sorted(section_stats.keys()),

    "section_stats":
        section_stats,

    "monthly_stats":
        monthly_stats,

    "area_sensitivity":
        area_sensitivity,

    "overall_median":
        overall_median,

    "normal_max_temp":
        normal_max_temp,

    "heat_sensitivity":
        heat_sensitivity,

    "version":
        "2.0"
}


joblib.dump(
    model_data,
    MODEL_PATH
)


print("")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "Sections:",
    len(section_stats)
)

print(
    "Heat sensitivity:",
    round(heat_sensitivity, 4)
)

print(
    "Model:",
    MODEL_PATH
)

print("")
print("Training complete.")