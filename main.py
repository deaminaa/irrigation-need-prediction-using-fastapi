from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "irrigation_model.pkl"


# ============================================================
# 2. LOAD MODEL ARTIFACT
# ============================================================

try:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
except Exception as exc:
    raise RuntimeError(f"Failed to load model artifact: {exc}")


model = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]
CATEGORICAL_FEATURES = artifact["categorical_features"]
THRESHOLDS = artifact["thresholds"]
CLASSES = artifact["classes"]


# ============================================================
# 3. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Irrigation Need Prediction API",
    description="Predicts irrigation requirement as Low, Medium, or High.",
    version="1.0.0",
)


# ============================================================
# 4. REQUEST SCHEMA
# ============================================================

class IrrigationInput(BaseModel):
    Soil_Type: str = Field(..., examples=["Loamy"])
    Soil_pH: float = Field(..., examples=[6.5])
    Soil_Moisture: float = Field(..., examples=[32.58])
    Electrical_Conductivity: float = Field(..., examples=[1.74])

    Temperature_C: float = Field(..., examples=[26.5])
    Humidity: float = Field(..., examples=[61.5])
    Rainfall_mm: float = Field(..., examples=[1200.0])
    Wind_Speed_kmh: float = Field(..., examples=[10.0])

    Crop_Type: str = Field(..., examples=["Wheat"])
    Crop_Growth_Stage: str = Field(..., examples=["Vegetative"])
    Season: str = Field(..., examples=["Rabi"])

    Irrigation_Type: str = Field(..., examples=["Drip"])
    Water_Source: str = Field(..., examples=["River"])
    Mulching_Used: str = Field(..., examples=["Yes"])

    Previous_Irrigation_mm: float = Field(..., examples=[50.0])


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreates the feature engineering used when the model
    was trained.
    """

    df = df.copy()

    # Earlier interaction features
    df["Temp_Wind_Interaction"] = (
        df["Temperature_C"] * df["Wind_Speed_kmh"]
    )

    df["Moisture_Temp_Interaction"] = (
        df["Soil_Moisture"] * df["Temperature_C"]
    )

    # Main engineered features
    df["ET_proxy"] = (
        df["Temperature_C"]
        * df["Wind_Speed_kmh"]
        * (1 - df["Humidity"] / 100)
    )

    df["Water_Deficit"] = (
        df["Rainfall_mm"] - df["ET_proxy"] * 2.5
    )

    df["Moisture_Rainfall_Ratio"] = (
        df["Soil_Moisture"] / (df["Rainfall_mm"] + 1e-3)
    )

    df["PrevIrrig_Impact"] = (
        df["Previous_Irrigation_mm"]
        * (1 + df["Soil_Moisture"] / 100)
    )

    soil_capacity = {
        "Sandy": 0.5,
        "Loamy": 1.0,
        "Silt": 1.2,
        "Clay": 1.5,
    }

    df["Soil_Capacity"] = df["Soil_Type"].map(soil_capacity)

    df["Moisture_vs_Capacity"] = (
        df["Soil_Moisture"]
        / (df["Soil_Capacity"] + 1e-3)
    )

    df["Temp_Deviation"] = np.abs(
        df["Temperature_C"] - 25
    )

    df["Temp_Effectiveness"] = np.maximum(
        0,
        30 - np.abs(df["Temperature_C"] - 25)
    )

    df["Irrig_Efficiency"] = (
        df["Previous_Irrigation_mm"]
        / (df["ET_proxy"] + 1)
    )

    df["Wind_Humidity"] = (
        df["Wind_Speed_kmh"]
        * (1 - df["Humidity"] / 100)
    )

    stage_order = {
        "Sowing": 0,
        "Vegetative": 1,
        "Flowering": 2,
        "Harvest": 3,
    }

    df["Stage_Ordinal"] = (
        df["Crop_Growth_Stage"].map(stage_order)
    )

    df["Moisture_x_Stage"] = (
        df["Soil_Moisture"] * df["Stage_Ordinal"]
    )

    df.drop(columns=["Stage_Ordinal"], inplace=True)

    df["Crop_Stage_Combo"] = (
        df["Crop_Type"].astype(str)
        + "_"
        + df["Crop_Growth_Stage"].astype(str)
    )

    df["Humidity_Temp_Interaction"] = (
        df["Humidity"]
        * df["Temperature_C"]
        / 100
    )

    df["Rainfall_SoilMoisture"] = (
        df["Rainfall_mm"]
        * df["Soil_Moisture"]
        / 100
    )

    return df


# ============================================================
# 6. PREPARE MODEL INPUT
# ============================================================

def prepare_features(data: IrrigationInput) -> pd.DataFrame:
    """
    Converts the API request into the exact feature layout
    expected by the trained model.
    """

    df = pd.DataFrame([data.model_dump()])

    # Feature engineering
    df = add_features(df)

    # Keep ONLY features used by the saved model
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    df = df[FEATURE_COLUMNS].copy()

    # --------------------------------------------------------
    # Restore categorical dtype.
    #
    # LightGBM stores the training categorical metadata inside
    # the underlying Booster. We use it when available.
    # --------------------------------------------------------

    booster = model.booster_

    pandas_categorical = getattr(
        booster,
        "pandas_categorical",
        None,
    )

    if pandas_categorical is not None:
        # Categorical columns are stored in the same order
        # as the categorical features used during training.
        for col, categories in zip(
            CATEGORICAL_FEATURES,
            pandas_categorical,
        ):
            if col in df.columns:
                df[col] = pd.Categorical(
                    df[col],
                    categories=categories,
                )
    else:
        # Fallback
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")

    return df


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Irrigation Need Prediction API is running",
        "model": "LightGBM",
        "classes": ["Low", "Medium", "High"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ============================================================
# 8. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: IrrigationInput):

    try:
        # Prepare model input
        X = prepare_features(data)

        # Probability prediction
        probabilities = model.predict_proba(X)[0]

        # ----------------------------------------------------
        # Apply stored thresholds
        # ----------------------------------------------------

        adjusted_probabilities = probabilities.copy()

        # Your actual artifact currently stores 0.5 / 0.5
        high_threshold = float(
            THRESHOLDS.get("high", 0.5)
        )

        medium_threshold = float(
            THRESHOLDS.get("medium", 0.5)
        )

        adjusted_probabilities[2] *= (
            high_threshold / 0.5
        )

        adjusted_probabilities[1] *= (
            medium_threshold / 0.5
        )

        adjusted_probabilities = (
            adjusted_probabilities
            / adjusted_probabilities.sum()
        )

        predicted_class = int(
            np.argmax(adjusted_probabilities)
        )

        label_map = {
            0: "Low",
            1: "Medium",
            2: "High",
        }

        prediction = label_map.get(
            predicted_class,
            str(predicted_class),
        )

        return {
            "prediction": prediction,
            "class_id": predicted_class,
            "probabilities": {
                "Low": round(
                    float(adjusted_probabilities[0]),
                    6,
                ),
                "Medium": round(
                    float(adjusted_probabilities[1]),
                    6,
                ),
                "High": round(
                    float(adjusted_probabilities[2]),
                    6,
                ),
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )