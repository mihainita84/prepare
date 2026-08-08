from __future__ import annotations

import requests
import pandas as pd

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
]

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
]


def fetch_weather(lat: float, lon: float, forecast_days: int = 3) -> tuple[dict, pd.DataFrame]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(CURRENT_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": forecast_days,
        "timezone": "Europe/Bucharest",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    current = data.get("current", {})
    hourly = pd.DataFrame(data.get("hourly", {}))
    if not hourly.empty and "time" in hourly.columns:
        hourly["time"] = pd.to_datetime(hourly["time"])
    return current, hourly


def portal_hazard_score(current: dict, ndmi: float | None = None) -> tuple[float, str, dict]:
    """Transparent screening score, deliberately NOT the official Canadian FWI."""
    t = float(current.get("temperature_2m") or 0)
    rh = float(current.get("relative_humidity_2m") or 100)
    wind = float(current.get("wind_speed_10m") or 0)
    vpd = float(current.get("vapour_pressure_deficit") or 0)
    sm = float(current.get("soil_moisture_0_to_1cm") or 0.45)
    rain = float(current.get("precipitation") or 0)

    temp_s = min(max((t - 15) / 25, 0), 1)
    rh_s = min(max((60 - rh) / 45, 0), 1)
    wind_s = min(max(wind / 40, 0), 1)
    vpd_s = min(max(vpd / 3.0, 0), 1)
    soil_s = min(max((0.38 - sm) / 0.30, 0), 1)
    rain_s = 1.0 if rain <= 0.05 else max(0.0, 1.0 - rain / 4.0)

    terms = {
        "temperature": temp_s,
        "low_humidity": rh_s,
        "wind": wind_s,
        "vpd": vpd_s,
        "surface_soil_dryness": soil_s,
        "no_recent_rain": rain_s,
    }
    weights = {
        "temperature": 0.15,
        "low_humidity": 0.20,
        "wind": 0.17,
        "vpd": 0.18,
        "surface_soil_dryness": 0.20,
        "no_recent_rain": 0.10,
    }

    if ndmi is not None:
        veg_dry = min(max((0.35 - ndmi) / 0.6, 0), 1)
        terms["vegetation_dryness_ndmi"] = veg_dry
        # reduce existing contribution slightly and add vegetation moisture
        score = 100 * (0.85 * sum(terms[k] * weights[k] for k in weights) + 0.15 * veg_dry)
    else:
        score = 100 * sum(terms[k] * weights[k] for k in weights)

    if score < 25:
        label = "Low"
    elif score < 45:
        label = "Moderate"
    elif score < 65:
        label = "High"
    elif score < 80:
        label = "Very high"
    else:
        label = "Extreme"
    return round(score, 1), label, terms
