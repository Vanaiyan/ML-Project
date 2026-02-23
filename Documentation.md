# Predicting Daily Precipitation Levels in Sri Lankan Cities for Flood Risk Assessment

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Methodology](#2-methodology)
   - 2.1 Dataset Description
   - 2.2 Feature Selection
   - 2.3 Data Preprocessing
   - 2.4 Algorithm Selection
   - 2.5 Model Training Strategy
   - 2.6 Hyperparameter Tuning
3. [Results](#3-results)
   - 3.1 Performance Metrics
   - 3.2 Model Comparison
   - 3.3 Visual Analysis
4. [Interpretation](#4-interpretation)
   - 4.1 Feature Importance
   - 4.2 SHAP Analysis
   - 4.3 Partial Dependence Plots
   - 4.4 Domain Alignment
5. [Discussion](#5-discussion)
   - 5.1 Model Limitations
   - 5.2 Data Quality Concerns
   - 5.3 Bias and Fairness
   - 5.4 Ethical Implications
   - 5.5 Future Work
6. [Front-End Integration](#6-front-end-integration)
7. [References](#7-references)

---

## 1. Problem Description

### 1.1 Background

Flooding is one of the most frequent and destructive natural disasters in Sri Lanka. Driven by intense monsoon rainfall and tropical weather systems, flood events routinely cause significant damage to infrastructure, agriculture, and human life. The country experiences two major monsoon seasons — the Southwest monsoon (May–September) and the Northeast monsoon (October–January) — both of which bring sustained periods of heavy rainfall to different geographic regions of the island.

Accurate prediction of daily precipitation levels is a critical prerequisite for effective flood risk assessment. With reliable precipitation forecasts, disaster management authorities can issue timely warnings, allocate emergency resources, and minimize the impact of flood events on vulnerable communities.

### 1.2 Objective

The objective of this project is to build a machine learning model that predicts **daily total precipitation (in millimeters)** across 30 Sri Lankan cities using meteorological, geographic, and temporal features. The predicted precipitation values can be classified into flood risk categories (Low, Moderate, High, Very High) to support practical decision-making.

### 1.3 Real-World Relevance

- **Early warning systems**: High precipitation predictions can trigger alerts for flood-prone regions.
- **Disaster resource allocation**: Authorities can pre-position emergency supplies and personnel based on predicted rainfall intensity.
- **Agricultural planning**: Farmers can make informed decisions about planting, irrigation, and harvest timing.
- **Urban infrastructure**: City planners can assess drainage system adequacy and plan maintenance schedules around high-rainfall periods.

---

## 2. Methodology

### 2.1 Dataset Description

| Property | Details |
|---|---|
| **Data Source** | Open-Meteo Historical Weather API (publicly available, open-source) |
| **Geographic Coverage** | 30 cities across Sri Lanka |
| **Temporal Coverage** | January 1, 2010 — June 17, 2023 |
| **Total Records** | 147,480 daily observations |
| **Original Columns** | 24 |
| **Ethical Considerations** | No personal or sensitive data; entirely public weather records |

The dataset contains daily weather observations for 30 Sri Lankan cities including Colombo, Kandy, Galle, Jaffna, Trincomalee, Ratnapura, Badulla, and others spanning both the wet zone (southwestern Sri Lanka) and the dry zone (northern and eastern regions). Each record includes temperature readings, apparent temperature, solar radiation, wind measurements, evapotranspiration, precipitation totals, and geographic metadata.

### 2.2 Feature Selection

#### Dependent Variable (Target)

- **`precipitation_sum`** — Total daily precipitation in millimeters (continuous variable, regression target)

Key statistics of the target variable:
- Mean: 5.98 mm
- Maximum: 338.8 mm
- Zero-precipitation days: 18.1% of all records
- The distribution is right-skewed, with the majority of days having low to moderate precipitation and occasional extreme rainfall events.

#### Independent Variables (20 Features)

| Category | Feature | Description |
|---|---|---|
| **Meteorological** | `temperature_2m_max` | Daily maximum temperature at 2m (°C) |
| | `temperature_2m_min` | Daily minimum temperature at 2m (°C) |
| | `temperature_2m_mean` | Daily mean temperature at 2m (°C) |
| | `apparent_temperature_max` | Maximum "feels-like" temperature (°C) — encodes humidity |
| | `apparent_temperature_min` | Minimum "feels-like" temperature (°C) |
| | `apparent_temperature_mean` | Mean "feels-like" temperature (°C) |
| | `shortwave_radiation_sum` | Total shortwave solar radiation (MJ/m²) — inversely related to cloud cover |
| | `windspeed_10m_max` | Maximum wind speed at 10m height (km/h) |
| | `windgusts_10m_max` | Maximum wind gusts at 10m height (km/h) |
| | `et0_fao_evapotranspiration` | FAO reference evapotranspiration (mm) — atmospheric moisture demand |
| **Geographic** | `latitude` | City latitude |
| | `longitude` | City longitude |
| | `elevation` | City elevation above sea level (m) |
| | `city_encoded` | Label-encoded city identifier |
| **Temporal** | `month` | Month of year (1–12) |
| | `day_of_year` | Day of year (1–366) |
| | `year` | Calendar year |
| | `day_length_hours` | Daylight duration derived from sunrise and sunset times |
| **Engineered** | `wind_dir_sin` | Sine of wind direction (circular encoding) |
| | `wind_dir_cos` | Cosine of wind direction (circular encoding) |

#### Excluded Variables and Rationale

| Feature | Reason for Exclusion |
|---|---|
| `rain_sum` | Identical to `precipitation_sum` when `snowfall_sum` = 0 — **direct data leakage** |
| `snowfall_sum` | Constant value of 0 across all records — zero variance, no predictive information |
| `precipitation_hours` | Number of hours with precipitation on the same day — **direct data leakage** |
| `weathercode` | Encodes the observed weather condition (rain/drizzle codes) — **data leakage** |
| `sunrise` / `sunset` | Raw timestamp strings; replaced by the derived `day_length_hours` feature |
| `country` | Always "Sri Lanka" — zero variance |
| `time` | Raw date string; replaced by engineered temporal features (`month`, `day_of_year`, `year`) |
| `winddirection_10m_dominant` | Raw degrees; replaced by circular `wind_dir_sin` and `wind_dir_cos` encoding |

The exclusion of `rain_sum`, `precipitation_hours`, and `weathercode` is critical. These variables directly encode information about precipitation that occurred on the same day and would cause the model to trivially "predict" the target from leaked information rather than learning genuine meteorological patterns.

### 2.3 Data Preprocessing

The following preprocessing steps were applied:

1. **Date parsing and temporal feature engineering**: The `time` column was parsed as a datetime object. Three temporal features were extracted: `month`, `day_of_year`, and `year`. These capture monsoon seasonality and long-term climate trends.

2. **Day length calculation**: The `sunrise` and `sunset` timestamps were parsed and differenced to compute `day_length_hours`, representing the duration of daylight.

3. **Circular encoding of wind direction**: Wind direction in degrees (0°–360°) is a circular variable where 0° and 360° are identical. To preserve this circularity, the direction was encoded as `sin(direction)` and `cos(direction)`.

4. **Label encoding of city**: The categorical `city` column was transformed to integer codes using scikit-learn's `LabelEncoder`.

5. **Missing value check**: No missing values were found in any column — no imputation was required.

6. **Feature scaling**: Standardization or normalization was **not applied**. XGBoost is a tree-based algorithm that splits on feature thresholds; it is invariant to monotonic transformations of features and does not require scaled inputs.

### 2.4 Algorithm Selection: XGBoost

**XGBoost (eXtreme Gradient Boosting)** was selected as the primary algorithm for this project. XGBoost is a scalable, optimized implementation of gradient-boosted decision trees that builds an ensemble of weak learners (shallow decision trees) sequentially, where each new tree is trained to correct the residual errors of the existing ensemble.

#### Key Characteristics

- **Gradient boosting framework**: Minimizes a differentiable loss function (squared error for regression) by iteratively fitting trees to the negative gradient of the loss.
- **Built-in regularization**: L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties on leaf weights prevent overfitting — a significant advantage over standard decision trees.
- **Column and row subsampling**: Each tree is trained on a random subset of features and samples, reducing variance and improving generalization.
- **Native handling of missing values**: The algorithm learns optimal split directions for missing data automatically.
- **Computational efficiency**: Parallel tree construction and cache-optimized data structures make XGBoost scalable to large datasets.

#### Comparison with Standard Algorithms

| Aspect | Linear Regression | Decision Tree | k-NN | XGBoost |
|---|---|---|---|---|
| **Model type** | Single linear model | Single tree | Instance-based | Ensemble of boosted trees |
| **Non-linearity** | Cannot capture | Captures but overfits | Local averaging | Captures via additive trees |
| **Feature interactions** | Must be manually specified | Captured via splits | Implicit via distance | Captured through tree depth |
| **Regularization** | Ridge/Lasso only | Pruning only | k selection | L1 + L2 + complexity penalties |
| **Skewed targets** | Assumes normality | Moderate handling | Moderate | Robust (non-parametric) |
| **Scalability** | Good | Good | Poor at scale | Excellent |

#### Justification for This Task

1. **Non-linear relationships**: Precipitation depends on complex, non-linear interactions between temperature, humidity, wind patterns, and geography that linear models cannot capture without extensive manual feature engineering.
2. **Skewed target distribution**: 18.1% of days have zero precipitation while extreme events exceed 300 mm. XGBoost handles this skewed distribution more robustly than linear approaches.
3. **Mixed feature types**: The dataset contains continuous, cyclical, and categorical features — tree-based methods handle this heterogeneity naturally.
4. **Regularization**: With 147,480 records and 20 features, regularized boosting generalizes well without overfitting.
5. **Explainability**: XGBoost integrates seamlessly with SHAP (SHapley Additive exPlanations) for both global and local model explanations.

### 2.5 Model Training Strategy

A **temporal split** was used rather than random splitting to prevent future data from leaking into training — this simulates a realistic deployment scenario where the model is trained on historical data and used to predict future precipitation.

| Split | Records | Percentage | Approximate Period |
|---|---|---|---|
| **Training** | 117,984 | 80% | 2010 – mid-2020 |
| **Validation** | 14,748 | 10% | mid-2020 – mid-2021 |
| **Test** | 14,748 | 10% | mid-2021 – June 2023 |

The validation set was used for hyperparameter tuning. The test set was held out entirely until final evaluation to provide an unbiased estimate of model performance.

Two baseline models were trained for comparison:
- **Linear Regression**: A simple parametric baseline to establish the floor of performance.
- **Random Forest** (200 trees, max depth 15): A strong ensemble baseline to contextualize XGBoost's improvement.

### 2.6 Hyperparameter Tuning

Hyperparameter optimization was performed using **Optuna**, a Bayesian optimization framework based on the Tree-structured Parzen Estimator (TPE) sampler. Optuna was run for 50 trials, optimizing on validation set RMSE.

The following hyperparameters were tuned:

| Hyperparameter | Search Range | Best Value |
|---|---|---|
| `n_estimators` | 200 – 1,000 (step 100) | 900 |
| `max_depth` | 3 – 10 | 8 |
| `learning_rate` | 0.01 – 0.3 (log scale) | 0.021 |
| `subsample` | 0.6 – 1.0 | 0.695 |
| `colsample_bytree` | 0.5 – 1.0 | 0.700 |
| `reg_alpha` (L1) | 0.001 – 10.0 (log scale) | 0.001 |
| `reg_lambda` (L2) | 0.001 – 10.0 (log scale) | 0.278 |
| `min_child_weight` | 1 – 10 | 9 |

The best configuration uses a large number of trees (900) with a low learning rate (0.021), deep trees (depth 8), and moderate subsampling (~70% of rows and columns per tree). The relatively high `min_child_weight` (9) prevents the model from fitting to noise in small leaf nodes.

---

## 3. Results

### 3.1 Performance Metrics

All metrics are reported on the held-out **test set** (14,748 samples from mid-2021 to June 2023).

| Model | RMSE (mm) | MAE (mm) | R² |
|---|---|---|---|
| Linear Regression | 6.464 | 4.166 | 0.4272 |
| Random Forest | 6.544 | 3.251 | 0.4130 |
| **XGBoost (Tuned)** | **6.354** | **3.123** | **0.4465** |

**Metric definitions:**
- **RMSE (Root Mean Squared Error)**: Square root of the average squared prediction error. Penalizes large errors more heavily. Lower is better.
- **MAE (Mean Absolute Error)**: Average absolute prediction error in millimeters. More interpretable than RMSE. Lower is better.
- **R² (Coefficient of Determination)**: Proportion of variance in precipitation explained by the model. Ranges from 0 to 1; higher is better.

### 3.2 Model Comparison

XGBoost (Tuned) achieves the best performance across all three metrics:

- **vs. Linear Regression**: XGBoost reduces RMSE by 1.7% (6.464 → 6.354), MAE by 25.0% (4.166 → 3.123), and improves R² by 4.5% (0.4272 → 0.4465). The large MAE improvement indicates XGBoost is substantially better at predicting typical (non-extreme) precipitation values.

- **vs. Random Forest**: XGBoost reduces RMSE by 2.9% (6.544 → 6.354), MAE by 3.9% (3.251 → 3.123), and improves R² by 8.1% (0.4130 → 0.4465). The boosting approach provides a meaningful improvement over bagging.

The R² value of 0.4465 indicates that the model explains approximately 44.7% of the variance in daily precipitation. While not exceptionally high, this is a reasonable result given the inherent stochasticity of daily precipitation — rainfall is fundamentally influenced by micro-scale atmospheric dynamics that cannot be captured from surface-level weather measurements alone.

### 3.3 Visual Analysis

The notebook includes the following visualizations:

1. **Actual vs. Predicted scatter plots**: For all three models, showing predicted precipitation on the y-axis and actual precipitation on the x-axis. The ideal model would produce points along the y = x diagonal. XGBoost's predictions cluster most tightly around this line.

2. **Residual distribution histograms**: The residual (actual − predicted) distributions are approximately centered at zero for all models, confirming no systematic bias. XGBoost exhibits the tightest residual distribution.

3. **Metric comparison bar charts**: Side-by-side comparison of RMSE, MAE, and R² across the three models, visually confirming XGBoost's superiority.

4. **Precipitation distribution plot**: Shows the right-skewed nature of the target variable, with most days between 0–10 mm and a long tail extending to 338.8 mm.

5. **Monthly average precipitation**: Reveals the bimodal monsoon pattern, with peaks corresponding to the Southwest and Northeast monsoons.

6. **Per-city average precipitation**: Shows geographic variation, with wet-zone cities (e.g., Ratnapura, Colombo) receiving more average rainfall than dry-zone cities (e.g., Jaffna, Hambantota).

---

## 4. Interpretation

### 4.1 Feature Importance (XGBoost Gain)

The built-in feature importance from XGBoost, ranked by total gain (improvement in the loss function contributed by a feature across all trees):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `et0_fao_evapotranspiration` | 0.1791 |
| 2 | `shortwave_radiation_sum` | 0.1116 |
| 3 | `temperature_2m_mean` | 0.0675 |
| 4 | `day_length_hours` | 0.0533 |
| 5 | `apparent_temperature_max` | 0.0520 |
| 6 | `apparent_temperature_min` | 0.0505 |
| 7 | `day_of_year` | 0.0493 |
| 8 | `month` | 0.0449 |
| 9 | `windgusts_10m_max` | 0.0434 |
| 10 | `windspeed_10m_max` | 0.0414 |

### 4.2 SHAP Analysis

Three types of SHAP plots were generated:

**SHAP Bar Plot (Global Importance)**: Shows the mean absolute SHAP value for each feature, representing the average magnitude of each feature's impact on predictions. This provides a model-agnostic importance ranking that complements the built-in gain measure.

**SHAP Beeswarm Plot (Global Pattern)**: Each dot represents one prediction. The x-axis shows the SHAP value (positive = increases predicted precipitation, negative = decreases it), and the color indicates the feature value (red = high, blue = low). Key observations:

- **`et0_fao_evapotranspiration`**: Low values (blue) push predictions strongly positive (more rain), while high values (red) push predictions negative (less rain). This aligns with the physical understanding that low evapotranspiration indicates saturated, humid atmospheric conditions conducive to rainfall.
- **`shortwave_radiation_sum`**: Low solar radiation (blue) strongly increases predicted precipitation — less sunshine means more cloud cover and higher rain likelihood.
- **Temperature features**: Higher apparent temperatures (which encode humidity via the heat index formula) tend to increase predicted precipitation, consistent with warm, humid tropical air masses.

**SHAP Dependence Plots**: Generated for the top 4 features, these show the functional relationship between each feature's value and its SHAP contribution. They reveal non-linear thresholds — for example, shortwave radiation below ~12 MJ/m² sharply increases predicted precipitation, while values above ~20 MJ/m² have a consistently negative effect.

**SHAP Waterfall Plot (Local Explanation)**: A single high-rainfall day was selected from the test set, and the waterfall plot shows how each feature pushed the prediction from the baseline (average precipitation) to the final predicted value. This demonstrates the model's ability to provide instance-level explanations for individual predictions.

### 4.3 Partial Dependence Plots

Partial Dependence Plots (PDPs) were generated for the top 4 features. PDPs show the marginal effect of a feature on the predicted outcome, averaged over all other features. Key findings:

- **`et0_fao_evapotranspiration`**: Precipitation decreases sharply as evapotranspiration increases from 0.5 to ~3 mm, then levels off. Very low evapotranspiration (<1 mm) is a strong indicator of rainfall.
- **`shortwave_radiation_sum`**: A clear negative relationship — predicted precipitation decreases as solar radiation increases, reflecting the cloud cover connection.
- **`temperature_2m_mean`**: A moderate positive relationship, consistent with convective rainfall being more common in warmer conditions.
- **`day_length_hours`**: Subtle seasonal signal; slightly longer days (associated with certain monsoon periods) show marginal differences in predicted precipitation.

### 4.4 Domain Alignment

The model's learned patterns align well with established meteorological knowledge of Sri Lanka:

1. **Evapotranspiration as the top predictor** is physically meaningful — low ET indicates high atmospheric humidity and saturated conditions, which are prerequisites for precipitation.

2. **Solar radiation's inverse relationship** with precipitation reflects the fundamental connection between cloud cover and rainfall — cloudy skies block solar radiation and are associated with precipitation events.

3. **Seasonal features** (`month`, `day_of_year`) capture Sri Lanka's bimodal monsoon pattern without being explicitly programmed.

4. **Geographic features** allow the model to differentiate between the wet zone (southwestern Sri Lanka, which receives 2,500+ mm annually) and the dry zone (northern and eastern regions, receiving less than 1,750 mm annually).

5. **Wind patterns** contribute to predictions by reflecting moisture-laden air masses — specific wind directions bring moisture from the Indian Ocean during monsoon periods.

---

## 5. Discussion

### 5.1 Model Limitations

**Same-day features, not true forecasting**: The model uses meteorological measurements from the same day (e.g., temperature, solar radiation) to predict precipitation. In a real-time operational setting, these values would not be available in advance. A true forecasting model would require lagged features (yesterday's weather to predict today's) or numerical weather prediction (NWP) model outputs as inputs. The current model should be understood as a **diagnostic tool** (understanding what drives precipitation) rather than a **prognostic tool** (forecasting future precipitation).

**No temporal autocorrelation modeling**: Each day is treated as an independent observation. In reality, weather patterns often persist over multiple days (e.g., multi-day monsoon events, stationary low-pressure systems). Incorporating lagged features, rolling averages, or time-series models could capture these dependencies.

**Extreme event underestimation**: The model is trained on a distribution where the vast majority of days have precipitation below 20 mm. Rare extreme events (>100 mm) are underrepresented in training data, and the model tends to underpredict these events. For flood risk assessment, accurately predicting these tail events is most critical.

**R² of 0.45**: While the model outperforms baselines, an R² of 0.45 means 55% of precipitation variance remains unexplained. Daily precipitation is inherently noisy and influenced by micro-scale atmospheric dynamics, convective processes, and topographic effects that cannot be fully captured from surface-level measurements at city scale.

### 5.2 Data Quality Concerns

**Single data source**: All data originates from the Open-Meteo API, which uses ERA5 reanalysis data. Reanalysis data is a model-generated best estimate, not direct observation. It may systematically underestimate localized heavy rainfall events or fail to capture fine-scale topographic effects, particularly in Sri Lanka's mountainous central highlands.

**Urban-centric sampling**: The 30 selected cities are primarily urban centers along major transportation corridors. Rural areas, mountainous terrain, and remote regions — where flooding may be equally or more devastating — are underrepresented in the dataset.

**Temporal boundary**: The dataset ends in June 2023. Climate change may alter precipitation patterns over time (shifting monsoon onset dates, intensifying extreme events), potentially degrading model performance if applied to future years without retraining.

### 5.3 Bias and Fairness

**Geographic prediction bias**: Model accuracy likely varies across cities. Cities with weather patterns similar to the training distribution (e.g., Colombo, which contributes the most records in terms of representativeness) may receive better predictions than cities with unique microclimates (e.g., Hatton at 1,281 m elevation).

**No socioeconomic context**: The model predicts precipitation quantity, not flood impact. A moderate rainfall prediction for a city with poor drainage infrastructure may cause more flooding than a heavy rainfall prediction for a city with robust flood defenses. Any deployment must integrate precipitation predictions with vulnerability assessments.

### 5.4 Ethical Implications

**False sense of security**: If the model underpredicts an extreme rainfall event, users relying on it may fail to take adequate precautions. Any deployment must clearly communicate prediction uncertainty and model limitations. The model should complement — not replace — official meteorological forecasts.

**Resource allocation risks**: If used to prioritize disaster preparedness resources, systematically biased predictions (e.g., better predictions for western cities than eastern cities) could lead to inequitable distribution of aid and emergency response.

**Data transparency**: The dataset consists entirely of publicly available weather records. No personal or sensitive information is used, and no privacy concerns arise from its use.

### 5.5 Future Work

1. **Lagged features**: Incorporate previous-day and multi-day rolling averages of temperature, humidity, and pressure to move toward true forecasting capability.
2. **Probabilistic predictions**: Replace point predictions with prediction intervals (e.g., using quantile regression or conformal prediction) to communicate uncertainty.
3. **Extreme value modeling**: Apply specialized techniques for predicting tail events — the most important component for flood risk assessment.
4. **Spatial features**: Incorporate terrain data (slope, aspect, distance to coast) and upstream catchment information for more physically meaningful flood risk assessment.
5. **Model retraining pipeline**: Establish automated retraining as new data becomes available to maintain model accuracy as climate patterns evolve.

---

## 6. Front-End Integration

A **Streamlit web application** (`app.py`) was developed to make the trained model accessible to non-technical users. The application provides:

### Features

1. **City selection**: A dropdown menu with all 30 Sri Lankan cities. Selecting a city automatically populates the geographic features (latitude, longitude, elevation).

2. **Date picker**: Allows selection of any date, from which temporal features (month, day of year, year) are automatically derived.

3. **Weather parameter inputs**: Sliders and number inputs for all meteorological features (temperature, apparent temperature, solar radiation, wind speed, wind gusts, evapotranspiration, wind direction, day length) with ranges based on the training data distribution.

4. **Precipitation prediction display**: Shows the predicted daily precipitation in millimeters as a prominent metric.

5. **Flood risk indicator**: A color-coded risk level based on predicted precipitation:
   - **Low** (green): < 2.5 mm — Minimal precipitation expected
   - **Moderate** (orange): 2.5–15 mm — Moderate rainfall
   - **High** (red): 15–50 mm — Heavy rain, potential for localized flooding
   - **Very High** (dark red): > 50 mm — Extreme rainfall, significant flood risk

6. **SHAP waterfall explanation**: For every prediction, a SHAP waterfall plot is generated showing how each feature contributed to that specific prediction — providing transparency and building user trust.

### Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 7. References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

2. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

3. Open-Meteo. (2023). Historical Weather API. https://open-meteo.com/en/docs/historical-weather-api

4. Department of Meteorology, Sri Lanka. Climatological data and monsoon patterns. http://www.meteo.gov.lk/

5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.

6. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

*Project developed as part of a Machine Learning assignment. All data used is publicly available and contains no personal or sensitive information.*
