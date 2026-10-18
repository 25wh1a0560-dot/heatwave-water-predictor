/* WHY THIS AREA */

document.getElementById(
    "modalWhyRisk"
).textContent =
    result.risk;


document.getElementById(
    "modalTemperature"
).textContent =
    document.getElementById(
        "max_temperature"
    ).value + "°C";


document.getElementById(
    "modalHumidity"
).textContent =
    document.getElementById(
        "humidity"
    ).value + "%";


document.getElementById(
    "modalWhyStress"
).textContent =
    Number(
        result.heat_stress
    ).toFixed(2);


document.getElementById(
    "modalWhySensitivity"
).textContent =
    Number(
        result.heat_sensitivity
    ).toFixed(3);


document.getElementById(
    "modalWhyIncrease"
).textContent =
    Number(
        result.increase
    ).toFixed(2) + "%";


let whyText =
    "The forecast heat conditions create increased water-demand pressure in this area.";

if (result.risk === "HIGH") {

    whyText =
        "High heat stress combined with this area's heat sensitivity produces a significant predicted increase in water demand.";

}

else if (result.risk === "MEDIUM") {

    whyText =
        "The forecast heat conditions create moderate demand pressure, so additional water supply preparation is recommended.";

}

else {

    whyText =
        "The forecast conditions produce relatively low demand pressure, so normal monitoring is sufficient.";

}


document.getElementById(
    "modalWhyText"
).textContent =
    whyText;