# 💧 Irrigation Need Prediction

An end-to-end machine learning application that predicts irrigation requirements as **Low, Medium, or High** using environmental, soil, crop, and irrigation-related features.

The trained **LightGBM** model is exposed through a **FastAPI** backend and a **Streamlit** frontend, with the complete application packaged into a **Docker** container for reproducible deployment.

---

## 🚀 Project Overview

Efficient irrigation is important for reducing unnecessary water usage while maintaining suitable conditions for crop growth.

This project builds a multiclass machine learning system that predicts irrigation needs from field and crop conditions such as:

- Soil type and soil moisture
- Soil pH and electrical conductivity
- Temperature and humidity
- Rainfall and wind speed
- Crop type and growth stage
- Season
- Irrigation type
- Water source
- Mulching
- Previous irrigation amount

The final system provides predictions through an interactive web interface and a REST API.

---

## 🧠 Machine Learning

### Model

The final model is a **LightGBM multiclass classifier** trained to predict three irrigation-need classes:

```text
Low
Medium
High
