import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. LOAD MODEL
# ==========================================

model_data = joblib.load(
    "model/water_demand_model.pkl"
)

weights = model_data["weights"]
mean = model_data["mean"]
std = model_data["std"]
features = model_data["features"]
sections = model_data["sections"]

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/hyderabad_2022_water_weather_training_data.csv"
)

df["date"] = pd.to_datetime(df["date"])

# ==========================================
# 3. GET USER INPUT
# ==========================================

print("\n====================================")
print(" HYDERABAD WATER DEMAND PREDICTOR")
print("====================================")

print("\nAvailable areas/sections:")
print("Example: BARKATPURA")
print("You can enter any section from the dataset.")

section = input("\nEnter section: ").strip().upper()

if section not in sections:
    print("\nSection not found.")
    print("Please check the spelling and try again.")
    exit()

temperature = float(
    input("Enter tomorrow's average temperature (°C): ")
)

max_temperature = float(
    input("Enter tomorrow's maximum temperature (°C): ")
)

humidity = float(
    input("Enter expected humidity (%): ")
)

rain = float(
    input("Enter expected rainfall (mm): ")
)

month = int(
    input("Enter forecast month (1-12): ")
)

# ==========================================
# 4. GET CURRENT WATER DEMAND
# ==========================================

section_data = df[
    df["section"].str.upper() == section
]

if len(section_data) == 0:
    print("No data found for this section.")
    exit()

latest = section_data.sort_values(
    "date"
).iloc[-1]

current_demand = float(
    latest["water_demand"]
)

# ==========================================
# 5. CREATE MONTH FEATURES
# ==========================================

month_sin = np.sin(
    2 * np.pi * month / 12
)

month_cos = np.cos(
    2 * np.pi * month / 12
)

# ==========================================
# 6. CREATE SECTION ENCODING
# ==========================================

section_encoded = np.zeros(
    len(sections)
)

section_index = sections.index(section)

section_encoded[section_index] = 1

# ==========================================
# 7. CREATE INPUT
# ==========================================

base_values = np.array([
    temperature,
    max_temperature,
    humidity,
    rain,
    current_demand,
    month_sin,
    month_cos
])

X = np.concatenate([
    base_values,
    section_encoded
])

# ==========================================
# 8. STANDARDIZE
# ==========================================

X_scaled = (
    X - mean
) / std

# ==========================================
# 9. PREDICT
# ==========================================

predicted_log = (
    weights[0]
    + np.dot(weights[1:], X_scaled)
)

predicted_demand = np.expm1(
    predicted_log
)

# ==========================================
# 10. CALCULATE DEMAND INCREASE
# ==========================================

demand_increase = (
    (predicted_demand - current_demand)
    / current_demand
) * 100

# ==========================================
# 11. RISK CLASSIFICATION
# ==========================================

if demand_increase < 10:
    risk = "LOW"
    action = "Monitor water demand normally."

elif demand_increase < 20:
    risk = "MEDIUM"
    action = "Prepare additional water supply."

else:
    risk = "HIGH"
    action = "Pre-position water tankers and temporary drinking-water stations."

# ==========================================
# 12. DISPLAY RESULT
# ==========================================

print("\n\n====================================")
print("          PREDICTION RESULT")
print("====================================")

print("Area / Section       :", section)
print("Tomorrow Avg Temp    :", temperature, "°C")
print("Tomorrow Max Temp    :", max_temperature, "°C")
print("Expected Humidity    :", humidity, "%")
print("Expected Rainfall    :", rain, "mm")

print("------------------------------------")

print(
    "Current Water Demand :",
    round(current_demand, 2)
)

print(
    "Predicted Water Demand:",
    round(predicted_demand, 2)
)

print(
    "Expected Demand Increase:",
    round(demand_increase, 2),
    "%"
)

print("------------------------------------")

print("RISK LEVEL           :", risk)
print("RECOMMENDED ACTION   :", action)

print("====================================")