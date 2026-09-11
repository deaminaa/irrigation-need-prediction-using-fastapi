from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


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
CLASSES = artifact["classes"]


# ============================================================
# 2a. VALIDATE ARTIFACT ASSUMPTIONS AT STARTUP
#
# Everything below is checked once, at import, so that a
# retrained or repacked artifact fails loudly here instead of
# producing plausible-looking but wrong predictions later.
# ============================================================

# The artifact records class ids (0/1/2) and never their names, so this list is
# the only surviving record of what each id means. Refuse to run if the ids stop
# matching it, otherwise every response would be mislabelled in silence.
CLASS_LABELS = ["Low", "Medium", "High"]

artifact_class_ids = [int(c) for c in CLASSES]
model_class_ids = [int(c) for c in model.classes_]

if artifact_class_ids != list(range(len(CLASS_LABELS))):
    raise RuntimeError(
        f"Artifact class ids {artifact_class_ids} do not match the label "
        f"order {CLASS_LABELS}; predictions would be mislabelled."
    )

if model_class_ids != artifact_class_ids:
    raise RuntimeError(
        f"Model classes_ {model_class_ids} disagree with artifact classes "
        f"{artifact_class_ids}."
    )

# The previous implementation rescaled probabilities by threshold / 0.5. At the
# stored 0.5 / 0.5 that is arithmetically a no-op, and at any other value it is
# not a decision rule either. It was removed rather than left as dead code, so
# refuse to start if the artifact ever carries real thresholds — silently
# ignoring them would be worse than not supporting them.
THRESHOLDS = artifact["thresholds"]

active_thresholds = {
    name: value
    for name, value in THRESHOLDS.items()
    if float(value) != 0.5
}

if active_thresholds:
    raise RuntimeError(
        f"Artifact carries non-default thresholds {active_thresholds} but no "
        "thresholding is implemented; predictions would ignore them."
    )

# LightGBM keeps the training categories inside the Booster, ordered to match
# CATEGORICAL_FEATURES. They are the only authoritative record of what the model
# will accept. Encoding against anything else silently produces wrong codes, so
# there is no fallback path here on purpose.
pandas_categorical = getattr(model.booster_, "pandas_categorical", None)

if not pandas_categorical:
    raise RuntimeError(
        "Booster is missing pandas_categorical; categorical inputs cannot be "
        "encoded to match training."
    )

# zip() would quietly truncate to the shorter of the two and leave some columns
# unencoded, so insist the pairing is exact.
if len(pandas_categorical) != len(CATEGORICAL_FEATURES):
    raise RuntimeError(
        f"Booster carries {len(pandas_categorical)} categorical category lists "
        f"but the artifact names {len(CATEGORICAL_FEATURES)} categorical "
        f"features; the two cannot be paired reliably."
    )

TRAINING_CATEGORIES = {
    column: list(categories)
    for column, categories in zip(CATEGORICAL_FEATURES, pandas_categorical)
}

# Crop_Stage_Combo is derived during feature engineering; the rest arrive in the
# request and are validated against these lists.
REQUEST_CATEGORICALS = [
    column
    for column in TRAINING_CATEGORIES
    if column != "Crop_Stage_Combo"
]

SOIL_CAPACITY = {
    "Sandy": 0.5,
    "Loamy": 1.0,
    "Silt": 1.2,
    "Clay": 1.5,
}

STAGE_ORDER = {
    "Sowing": 0,
    "Vegetative": 1,
    "Flowering": 2,
    "Harvest": 3,
}

# A category the model knows but these tables do not would map to NaN and still
# yield a confident prediction, so check the coverage up front.
for column, lookup in (
    ("Soil_Type", SOIL_CAPACITY),
    ("Crop_Growth_Stage", STAGE_ORDER),
):
    unmapped = set(TRAINING_CATEGORIES[column]) - set(lookup)

    if unmapped:
        raise RuntimeError(
            f"{column} categories {sorted(unmapped)} have no feature-"
            f"engineering entry; they would silently become NaN."
        )

# Crop_Stage_Combo is derived from two validated inputs, so every crop x stage
# pairing has to exist in training or the derived value becomes NaN for an
# otherwise perfectly valid request.
expected_combos = {
    f"{crop}_{stage}"
    for crop in TRAINING_CATEGORIES["Crop_Type"]
    for stage in TRAINING_CATEGORIES["Crop_Growth_Stage"]
}

missing_combos = expected_combos - set(TRAINING_CATEGORIES["Crop_Stage_Combo"])

if missing_combos:
    raise RuntimeError(
        f"Crop_Stage_Combo is missing {sorted(missing_combos)}; those "
        f"combinations would silently become NaN."
    )


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

    @model_validator(mode="after")
    def reject_untrained_categories(self):
        """
        An unrecognised category becomes NaN once encoded, which LightGBM reads
        as a missing value and still scores confidently. Reject it at the trust
        boundary so the caller gets a 422 instead of a plausible answer.
        """

        for column in REQUEST_CATEGORICALS:
            value = getattr(self, column)
            allowed = TRAINING_CATEGORIES[column]

            if value not in allowed:
                raise ValueError(
                    f"{column}={value!r} is not a category the model was "
                    f"trained on. Valid values: {allowed}"
                )

        return self


unknown_categoricals = (
    set(REQUEST_CATEGORICALS) - set(IrrigationInput.model_fields)
)

if unknown_categoricals:
    raise RuntimeError(
        f"Categorical features {sorted(unknown_categoricals)} are not request "
        f"fields, so they would never be validated."
    )


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

    df["Soil_Capacity"] = df["Soil_Type"].map(SOIL_CAPACITY)

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

    df["Stage_Ordinal"] = (
        df["Crop_Growth_Stage"].map(STAGE_ORDER)
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
    # Restore the categorical dtype exactly as it was during
    # training. Both the category lists and the request values
    # were validated at import and by the schema, so nothing
    # reaching this point can fall outside its category list.
    # --------------------------------------------------------

    for column, categories in TRAINING_CATEGORIES.items():
        if column in df.columns:
            df[column] = pd.Categorical(
                df[column],
                categories=categories,
            )

    return df


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Irrigation Need Prediction API is running",
        "model": "LightGBM",
        "classes": CLASS_LABELS,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/categories")
def categories():
    """
    Valid values for every categorical input, read off the model itself.

    The UI builds its dropdowns from this response so the two cannot drift
    apart as the model is retrained.
    """

    return {
        column: TRAINING_CATEGORIES[column]
        for column in REQUEST_CATEGORICALS
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

        predicted_class = int(np.argmax(probabilities))

        return {
            "prediction": CLASS_LABELS[predicted_class],
            "class_id": predicted_class,
            "probabilities": {
                label: round(float(probabilities[index]), 6)
                for index, label in enumerate(CLASS_LABELS)
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