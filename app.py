from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "water_demand_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model_data = joblib.load(MODEL_PATH)

    print("=" * 60)
    print("HYDERABAD WATER DEMAND PREDICTOR")
    print("=" * 60)

    print(
        "Model type:",
        model_data.get("model_type", "Unknown")
    )

    print(
        "Sections:",
        len(model_data.get("sections", []))
    )

    print("Model loaded successfully.")

except Exception as e:

    print("MODEL LOADING ERROR:")
    print(e)

    model_data = None


# ============================================================
# INPUT HELPERS
# ============================================================

def get_number(data, *names, default=None):

    for name in names:

        value = data.get(name)

        if value is not None and value != "":

            try:
                return float(value)
            except:
                pass

    if default is not None:
        return float(default)

    raise ValueError(
        "Missing input: " + ", ".join(names)
    )


def get_integer(data, *names, default=None):

    for name in names:

        value = data.get(name)

        if value is not None and value != "":

            try:
                return int(float(value))
            except:
                pass

    if default is not None:
        return int(default)

    raise ValueError(
        "Missing month input."
    )


# ============================================================
# SECTION HELPERS
# ============================================================

def clean_section_name(section):

    return str(section).strip().upper()


def get_section_index(section):

    sections = model_data["sections"]

    clean_sections = [
        clean_section_name(s)
        for s in sections
    ]

    section = clean_section_name(section)

    if section not in clean_sections:

        raise ValueError(
            f"Section '{section}' not found."
        )

    return clean_sections.index(section)


# ============================================================
# MAIN PREDICTION
# ============================================================

def make_prediction(
    section,
    avg_temperature,
    max_temperature,
    humidity,
    rainfall,
    month
):

    if model_data is None:

        raise ValueError(
            "Model could not be loaded."
        )


    section = clean_section_name(section)

    get_section_index(section)


    # --------------------------------------------------------
    # HISTORICAL SECTION DEMAND
    # --------------------------------------------------------

    section_stats = model_data[
        "section_stats"
    ][section]

    current_demand = float(
        section_stats["median_demand"]
    )


    if current_demand <= 0:

        current_demand = float(
            model_data["overall_median"]
        )


    # --------------------------------------------------------
    # AREA HEAT SENSITIVITY
    # --------------------------------------------------------

    raw_sensitivity = float(
        model_data[
            "area_sensitivity"
        ].get(section, 0.0)
    )


    # Convert historical sensitivity into a
    # controlled scenario sensitivity.
    #
    # Final range = 0.05 to 0.28

    sensitivity = 0.05 + (
        np.clip(
            raw_sensitivity,
            0.0,
            0.30
        )
        / 0.30
    ) * 0.23


    sensitivity = float(
        np.clip(
            sensitivity,
            0.05,
            0.28
        )
    )


    # --------------------------------------------------------
    # NORMAL MAXIMUM TEMPERATURE
    # --------------------------------------------------------

    normal_max_temp = float(
        model_data[
            "normal_max_temp"
        ]
    )


    # --------------------------------------------------------
    # TEMPERATURE STRESS
    #
    # 12.5°C above the historical normal maximum
    # corresponds to stress = 1.
    #
    # Example:
    # 31.5°C normal
    # 44°C forecast
    # difference = 12.5°C
    # stress = 1
    # --------------------------------------------------------

    temperature_stress = (
        float(max_temperature)
        - normal_max_temp
    ) / 12.5


    temperature_stress = max(
        0.0,
        temperature_stress
    )


    temperature_stress = min(
        1.0,
        temperature_stress
    )


    # --------------------------------------------------------
    # HUMIDITY STRESS
    #
    # Lower humidity slightly increases heat stress.
    # --------------------------------------------------------

    humidity_stress = (
        50.0 - float(humidity)
    ) / 20.0


    humidity_stress = np.clip(
        humidity_stress,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # RAINFALL STRESS
    #
    # Low rainfall slightly increases demand pressure.
    # --------------------------------------------------------

    rain_stress = (
        10.0 - float(rainfall)
    ) / 20.0


    rain_stress = np.clip(
        rain_stress,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # COMBINED HEAT STRESS
    # --------------------------------------------------------

    heat_stress = (

        0.80 * temperature_stress

        + 0.15 * humidity_stress

        + 0.05 * rain_stress
    )


    heat_stress = np.clip(
        heat_stress,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # HEAT-DRIVEN DEMAND INCREASE
    # --------------------------------------------------------

    increase_fraction = (
        sensitivity
        * heat_stress
    )


    increase_fraction = np.clip(
        increase_fraction,
        0.0,
        0.30
    )


    # --------------------------------------------------------
    # PREDICTED DEMAND
    # --------------------------------------------------------

    predicted_demand = (
        current_demand
        *
        (1.0 + increase_fraction)
    )


    increase_percent = (
        increase_fraction
        * 100.0
    )


    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if increase_percent >= 20:

        risk = "HIGH"

        action = (
            "Pre-position water tankers and "
            "temporary drinking-water stations."
        )

    elif increase_percent >= 10:

        risk = "MEDIUM"

        action = (
            "Prepare additional water supply "
            "and monitor demand closely."
        )

    else:

        risk = "LOW"

        action = (
            "Normal water supply is expected. "
            "Continue monitoring."
        )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "section":
            section,

        "current_demand":
            round(current_demand, 2),

        "predicted_demand":
            round(predicted_demand, 2),

        "increase":
            round(increase_percent, 2),

        "risk":
            risk,

        "action":
            action,

        "heat_stress":
            round(float(heat_stress), 3),

        "heat_sensitivity":
            round(float(sensitivity), 3),

        "temperature":
            float(avg_temperature),

        "max_temperature":
            float(max_temperature),

        "humidity":
            float(humidity),

        "rainfall":
            float(rainfall),

        "month":
            int(month)
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ALL PREDICTIONS
# ============================================================

@app.route(
    "/all_predictions",
    methods=["POST"]
)
def all_predictions():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        print("\nReceived dashboard request:")
        print(data)


        avg_temperature = get_number(
            data,
            "avg_temperature",
            "avgTemperature",
            "temperature"
        )


        max_temperature = get_number(
            data,
            "max_temperature",
            "maxTemperature",
            "maxTemp"
        )


        humidity = get_number(
            data,
            "humidity"
        )


        rainfall = get_number(
            data,
            "rainfall",
            "rain",
            "rain_mm",
            default=0
        )


        month = get_integer(
            data,
            "month",
            "forecast_month",
            "forecastMonth",
            default=5
        )


        if month < 1 or month > 12:

            raise ValueError(
                "Month must be between 1 and 12."
            )


        results = []


        for section in model_data["sections"]:

            try:

                result = make_prediction(

                    section,

                    avg_temperature,

                    max_temperature,

                    humidity,

                    rainfall,

                    month

                )

                results.append(result)

            except Exception as e:

                print(
                    "Skipping:",
                    section,
                    e
                )


        # Highest demand increase first

        results.sort(
            key=lambda x: x["increase"],
            reverse=True
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        high_count = sum(
            1
            for r in results
            if r["risk"] == "HIGH"
        )


        medium_count = sum(
            1
            for r in results
            if r["risk"] == "MEDIUM"
        )


        low_count = sum(
            1
            for r in results
            if r["risk"] == "LOW"
        )


        return jsonify({

            "success": True,

            "results": results,

            "summary": {

                "total_sections":
                    len(results),

                "high":
                    high_count,

                "medium":
                    medium_count,

                "low":
                    low_count
            }

        })


    except Exception as e:

        print(
            "PREDICTION ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        section = data.get(
            "section"
        )


        if not section:

            raise ValueError(
                "Missing section."
            )


        avg_temperature = get_number(
            data,
            "avg_temperature",
            "avgTemperature",
            "temperature"
        )


        max_temperature = get_number(
            data,
            "max_temperature",
            "maxTemperature",
            "maxTemp"
        )


        humidity = get_number(
            data,
            "humidity"
        )


        rainfall = get_number(
            data,
            "rainfall",
            "rain",
            default=0
        )


        month = get_integer(
            data,
            "month",
            "forecast_month",
            "forecastMonth",
            default=5
        )


        result = make_prediction(

            section,

            avg_temperature,

            max_temperature,

            humidity,

            rainfall,

            month

        )


        return jsonify({

            "success": True,

            "result": result

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if model_data is None:

        return jsonify({

            "status":
                "error",

            "message":
                "Model not loaded."

        })


    return jsonify({

        "status":
            "ok",

        "model":
            model_data.get(
                "model_type"
            ),

        "sections":
            len(
                model_data[
                    "sections"
                ]
            )

    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print("HYDERABAD WATER COMMAND CENTER")
    print("=" * 60)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )