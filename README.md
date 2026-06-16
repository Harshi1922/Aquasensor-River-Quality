# AquaSensor River Water Quality
## ML Prediction and Anomaly Detection System

**MSc Computing — Sheffield Hallam University**
**Module:** Research Skills for Computing (55-710248)

## Project
Machine learning system that:
1. Predicts dissolved oxygen 15 minutes ahead
2. Detects anomalies (actual DO vs predicted DO)
3. Explains alerts using SHAP
4. Shows everything on an interactive Streamlit dashboard

## Research Question
How effectively can a combined ML framework predict 
dissolved oxygen levels 15 minutes ahead and 
automatically detect anomalous events in AquaSensor 
river sensor data to support early identification 
of pollution incidents?

## Data Sources
- AquaSensor API — 3 sensors on River Derwent
  (temperature + DO every ~12 minutes)
- Environment Agency — station E01786A
  (river level + flow)
- Open-Meteo API — free weather data
  (air temperature, sunshine, rainfall, wind)

## Sensors
- sensor022 — Derwent 13 — Hathersage
- 941115    — Derwent 13-50 — Near reservoir
- 941205    — Derwent 21 — Grindleford

## Models
- Linear Regression (baseline)
- Random Forest
- XGBoost

## Primary Reference
Jarwar et al. (2025) — IoT digital twin River Derwent
https://shura.shu.ac.uk/36324/