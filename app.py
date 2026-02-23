import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

FEATURE_EXPLANATIONS = {
    "et0_fao_evapotranspiration": {
        "name": "Evapotranspiration",
        "high": "High evapotranspiration indicates dry atmospheric conditions with strong moisture demand, making rainfall less likely.",
        "low": "Low evapotranspiration signals a humid, saturated atmosphere — ideal conditions for precipitation.",
    },
    "shortwave_radiation_sum": {
        "name": "Solar Radiation",
        "high": "High solar radiation means clear skies with little cloud cover, reducing the chance of rain.",
        "low": "Low solar radiation indicates heavy cloud cover, which is strongly associated with rainfall.",
    },
    "temperature_2m_mean": {
        "name": "Mean Temperature",
        "high": "Higher temperatures can fuel convective activity, increasing the potential for rainfall in tropical climates.",
        "low": "Lower temperatures reduce convective energy, generally leading to less precipitation.",
    },
    "temperature_2m_max": {
        "name": "Max Temperature",
        "high": "A high maximum temperature can drive strong convection and thunderstorm development.",
        "low": "A lower maximum temperature suggests reduced convective potential.",
    },
    "temperature_2m_min": {
        "name": "Min Temperature",
        "high": "A warm overnight minimum indicates persistent humidity, supporting continued rainfall.",
        "low": "A cooler overnight minimum suggests drier conditions.",
    },
    "apparent_temperature_max": {
        "name": "Max Apparent Temperature",
        "high": "A very high 'feels-like' temperature signals extreme humidity — a key precursor to heavy rain.",
        "low": "A lower apparent temperature suggests drier air with less moisture available for precipitation.",
    },
    "apparent_temperature_min": {
        "name": "Min Apparent Temperature",
        "high": "High overnight apparent temperature means sustained humidity, keeping conditions ripe for rain.",
        "low": "Lower apparent temperature overnight suggests humidity is dropping.",
    },
    "apparent_temperature_mean": {
        "name": "Mean Apparent Temperature",
        "high": "Consistently high 'feels-like' temperature throughout the day reflects high humidity levels.",
        "low": "Lower mean apparent temperature indicates relatively drier conditions.",
    },
    "windspeed_10m_max": {
        "name": "Max Wind Speed",
        "high": "Strong winds can carry moisture-laden air masses inland, increasing rainfall potential.",
        "low": "Light winds suggest stable atmospheric conditions with less moisture transport.",
    },
    "windgusts_10m_max": {
        "name": "Max Wind Gusts",
        "high": "Intense wind gusts often accompany storm systems that bring heavy rainfall.",
        "low": "Calm conditions with weak gusts suggest stable, dry weather.",
    },
    "month": {
        "name": "Month of Year",
        "high": "This time of year historically corresponds to one of Sri Lanka's monsoon seasons.",
        "low": "This month typically falls in a drier inter-monsoon period.",
    },
    "day_of_year": {
        "name": "Day of Year",
        "high": "This part of the year aligns with seasonal rainfall patterns in the region.",
        "low": "This period tends to be drier based on historical weather patterns.",
    },
    "year": {
        "name": "Year",
        "high": "Recent years may show different precipitation trends due to climate variability.",
        "low": "Earlier years had different baseline precipitation patterns.",
    },
    "day_length_hours": {
        "name": "Day Length",
        "high": "Longer daylight hours can correlate with specific seasonal rainfall patterns.",
        "low": "Shorter days are associated with different monsoon periods.",
    },
    "wind_dir_sin": {
        "name": "Wind Direction (N-S component)",
        "high": "Winds from the south carry Indian Ocean moisture toward Sri Lanka.",
        "low": "Northerly wind components bring drier continental air.",
    },
    "wind_dir_cos": {
        "name": "Wind Direction (E-W component)",
        "high": "Easterly winds can bring moisture from the Bay of Bengal.",
        "low": "Westerly winds carry Arabian Sea moisture during the SW monsoon.",
    },
    "latitude": {
        "name": "Latitude (Location)",
        "high": "Northern Sri Lanka (dry zone) generally receives less annual rainfall.",
        "low": "Southern Sri Lanka (wet zone) typically experiences heavier and more frequent rainfall.",
    },
    "longitude": {
        "name": "Longitude (Location)",
        "high": "Eastern cities receive more rainfall during the NE monsoon (Oct–Jan).",
        "low": "Western coastal cities are more exposed to the SW monsoon (May–Sep).",
    },
    "elevation": {
        "name": "Elevation",
        "high": "Highland areas experience orographic rainfall as moist air rises over mountains.",
        "low": "Low-lying areas may receive less orographic enhancement of rainfall.",
    },
    "city_encoded": {
        "name": "City",
        "high": "This city's historical rainfall pattern influences the prediction.",
        "low": "This city's historical rainfall pattern influences the prediction.",
    },
}

st.set_page_config(
    page_title="Sri Lanka Precipitation Predictor",
    page_icon="🌧️",
    layout="wide",
)

CITY_GEO = {
    "Athurugiriya": (6.9, 79.9, 27.0),
    "Badulla": (7.1, 81.1, 652.0),
    "Bentota": (6.5, 80.0, 10.0),
    "Colombo": (7.0, 79.9, 16.0),
    "Galle": (6.1, 80.2, 15.0),
    "Gampaha": (7.1, 80.0, 15.0),
    "Hambantota": (6.2, 81.2, 12.0),
    "Hatton": (6.9, 80.6, 1281.0),
    "Jaffna": (9.7, 80.0, 5.0),
    "Kalmunai": (7.4, 81.8, 8.0),
    "Kalutara": (6.6, 80.0, 0.0),
    "Kandy": (7.3, 80.6, 510.0),
    "Kesbewa": (6.8, 79.9, 18.0),
    "Kolonnawa": (6.9, 79.9, 13.0),
    "Kurunegala": (7.5, 80.4, 124.0),
    "Mabole": (7.0, 79.9, 16.0),
    "Maharagama": (6.8, 79.9, 26.0),
    "Mannar": (8.9, 80.0, 6.0),
    "Matale": (7.6, 80.6, 376.0),
    "Matara": (6.0, 80.4, 7.0),
    "Moratuwa": (6.8, 79.9, 9.0),
    "Mount Lavinia": (6.9, 79.9, 10.0),
    "Negombo": (7.1, 79.9, 5.0),
    "Oruwala": (6.9, 80.0, 20.0),
    "Pothuhera": (7.4, 80.3, 125.0),
    "Puttalam": (8.0, 79.8, 0.0),
    "Ratnapura": (6.8, 80.3, 27.0),
    "Sri Jayewardenepura Kotte": (6.9, 79.9, 7.0),
    "Trincomalee": (8.6, 81.2, 7.0),
    "Weligama": (6.0, 80.4, 5.0),
}


@st.cache_resource
def load_artifacts():
    return joblib.load("model_artifacts.joblib")


artifacts = load_artifacts()
model = artifacts["model"]
feature_cols = artifacts["feature_cols"]
le_city = artifacts["label_encoder_city"]
city_names = artifacts["city_names"]

st.title("Sri Lanka Daily Precipitation Predictor")
st.markdown(
    "Predict daily precipitation levels across Sri Lankan cities and assess flood risk "
    "using an XGBoost model trained on historical weather data (2010–2023)."
)

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Input Weather Parameters")

    city = st.selectbox("City", city_names, index=city_names.index("Colombo"))
    lat, lon, elev = CITY_GEO[city]

    date = st.date_input("Date", value=pd.Timestamp("2023-06-01"))
    month = date.month
    day_of_year = date.timetuple().tm_yday
    year = date.year

    st.markdown("**Temperature (°C)**")
    c1, c2, c3 = st.columns(3)
    temp_max = c1.number_input("Max", 17.5, 37.5, 30.0, 0.5)
    temp_min = c2.number_input("Min", 12.0, 30.0, 24.0, 0.5)
    temp_mean = c3.number_input("Mean", 16.0, 32.0, 27.0, 0.5)

    st.markdown("**Apparent Temperature (°C)**")
    c1, c2, c3 = st.columns(3)
    app_max = c1.number_input("App Max", 18.5, 43.5, 34.0, 0.5)
    app_min = c2.number_input("App Min", 10.5, 35.0, 28.0, 0.5)
    app_mean = c3.number_input("App Mean", 16.0, 37.5, 30.0, 0.5)

    radiation = st.slider("Shortwave Radiation (MJ/m²)", 1.0, 29.0, 18.0, 0.5)
    wind_speed = st.slider("Max Wind Speed (km/h)", 2.0, 50.0, 15.0, 0.5)
    wind_gust = st.slider("Max Wind Gust (km/h)", 11.0, 92.0, 35.0, 0.5)
    et0 = st.slider("Evapotranspiration (mm)", 0.4, 8.0, 3.9, 0.1)
    wind_dir = st.slider("Wind Direction (°)", 0, 360, 180, 5)
    day_length = st.slider("Day Length (hours)", 11.5, 12.8, 12.1, 0.1)

with col_right:
    wind_dir_rad = np.deg2rad(wind_dir)
    city_encoded = le_city.transform([city])[0]

    input_values = [
        temp_max, temp_min, temp_mean,
        app_max, app_min, app_mean,
        radiation, wind_speed, wind_gust, et0,
        lat, lon, elev,
        month, day_of_year, year,
        day_length,
        np.sin(wind_dir_rad), np.cos(wind_dir_rad),
        city_encoded,
    ]

    input_df = pd.DataFrame([input_values], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    prediction = max(0.0, prediction)

    st.subheader("Prediction Result")

    if prediction < 2.5:
        risk_level, risk_color, risk_desc = "Low", "green", "Minimal precipitation expected."
    elif prediction < 15:
        risk_level, risk_color, risk_desc = "Moderate", "orange", "Moderate rainfall — stay aware."
    elif prediction < 50:
        risk_level, risk_color, risk_desc = "High", "red", "Heavy rain — potential for localized flooding."
    else:
        risk_level, risk_color, risk_desc = "Very High", "darkred", "Extreme rainfall — significant flood risk."

    m1, m2 = st.columns(2)
    m1.metric("Predicted Precipitation", f"{prediction:.1f} mm")
    m2.metric("Flood Risk Level", risk_level)

    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background-color:{risk_color};"
        f"color:white;font-size:16px;text-align:center;margin-bottom:16px;'>"
        f"<b>{risk_level} Risk</b> — {risk_desc}</div>",
        unsafe_allow_html=True,
    )

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)

    shap_series = pd.Series(shap_vals[0], index=feature_cols)
    base_value = explainer.expected_value

    # --- Human-readable XAI explanation (shown first) ---
    st.subheader("Why This Prediction?")

    top_k = 5
    top_factors = shap_series.abs().sort_values(ascending=False).head(top_k).index.tolist()

    increasing = []
    decreasing = []
    for feat in top_factors:
        sv = shap_series[feat]
        val = input_df[feat].values[0]
        info = FEATURE_EXPLANATIONS.get(feat, {"name": feat, "high": "", "low": ""})
        median_val = {
            "et0_fao_evapotranspiration": 3.9, "shortwave_radiation_sum": 18.5,
            "temperature_2m_mean": 26.2, "temperature_2m_max": 29.2, "temperature_2m_min": 23.9,
            "apparent_temperature_max": 34.1, "apparent_temperature_min": 27.7,
            "apparent_temperature_mean": 30.3, "windspeed_10m_max": 15.6,
            "windgusts_10m_max": 34.8, "month": 6, "day_of_year": 183, "year": 2016,
            "day_length_hours": 12.15, "wind_dir_sin": 0.0, "wind_dir_cos": 0.0,
            "latitude": 7.1, "longitude": 80.3, "elevation": 112.0, "city_encoded": 15,
        }.get(feat, 0)
        is_high = val >= median_val
        reason = info["high"] if is_high else info["low"]
        entry = (info["name"], abs(sv), reason)
        if sv > 0:
            increasing.append(entry)
        else:
            decreasing.append(entry)

    if prediction < 2.5:
        summary = (
            f"The model predicts **{prediction:.1f} mm** of precipitation for **{city}** — "
            f"a **dry day** with minimal rain expected. Here's why:"
        )
    elif prediction < 15:
        summary = (
            f"The model predicts **{prediction:.1f} mm** of precipitation for **{city}** — "
            f"a **moderately rainy day**. Here's what's driving this prediction:"
        )
    elif prediction < 50:
        summary = (
            f"The model predicts **{prediction:.1f} mm** of precipitation for **{city}** — "
            f"a **heavy rainfall day** with potential flood risk. Key factors:"
        )
    else:
        summary = (
            f"The model predicts **{prediction:.1f} mm** of precipitation for **{city}** — "
            f"an **extreme rainfall event** with significant flood risk. Key factors:"
        )

    st.markdown(summary)

    if increasing:
        st.markdown("**Factors increasing predicted rainfall:**")
        for name, strength, reason in sorted(increasing, key=lambda x: -x[1]):
            st.markdown(f"- **{name}**: {reason}")

    if decreasing:
        st.markdown("**Factors decreasing predicted rainfall:**")
        for name, strength, reason in sorted(decreasing, key=lambda x: -x[1]):
            st.markdown(f"- **{name}**: {reason}")

    st.info(
        f"**How to read this:** The model starts from a baseline of {base_value:.1f} mm "
        f"(the average daily precipitation across all training data). Each weather condition you entered "
        f"either pushes the prediction higher (more rain) or lower (less rain). "
        f"The SHAP chart below shows the exact numerical contribution of each factor, "
        f"while this section explains the meteorological reasoning."
    )

    # --- SHAP chart (shown below the explanation) ---
    st.subheader("SHAP Feature Contribution Chart")
    st.caption("Exact numerical impact of each feature on this prediction:")

    top_n = 15
    top_shap = shap_series.reindex(shap_series.abs().sort_values(ascending=False).head(top_n).index)
    top_shap = top_shap.sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#ff0051" if v > 0 else "#008bfb" for v in top_shap.values]
    ax.barh(range(len(top_shap)), top_shap.values, color=colors, edgecolor="none", height=0.6)

    for i, (feat, val) in enumerate(zip(top_shap.index, top_shap.values)):
        input_val = input_df[feat].values[0]
        label = f"{feat} = {input_val:.2f}"
        ax.text(-0.01 if val > 0 else 0.01, i, label,
                ha="right" if val > 0 else "left", va="center", fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
    ax.set_title(
        f"Base value: {base_value:.2f} mm  →  Prediction: {prediction:.2f} mm",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    red_patch = plt.Line2D([0], [0], color="#ff0051", lw=6, label="Pushes prediction UP")
    blue_patch = plt.Line2D([0], [0], color="#008bfb", lw=6, label="Pushes prediction DOWN")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Model: XGBoost Regressor trained on 147,480 daily weather records across 30 Sri Lankan cities (2010–2023). "
    "Data source: Open-Meteo Historical Weather API."
)
