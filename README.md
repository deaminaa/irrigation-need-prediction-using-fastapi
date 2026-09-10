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

The saved model artifact contains the trained model together with the feature configuration required for inference.

Feature Engineering

The prediction pipeline uses both raw agricultural features and engineered features, including:

Temperature × Wind Speed
Soil Moisture × Temperature
Evapotranspiration proxy
Water deficit
Moisture-to-rainfall ratio
Previous irrigation impact
Soil water-holding capacity
Moisture relative to soil capacity
Temperature deviation/effectiveness
Irrigation efficiency
Wind-humidity interaction
Moisture × crop growth stage
Crop × growth-stage combination
Humidity × temperature interaction
Rainfall × soil-moisture interaction

The API recreates these transformations during inference so that predictions use the same feature structure as the trained model.
🏗️ System Architecture
                   User
                     │
                     ▼
              Streamlit UI
            (irrigation.py)
                     │
                     │ HTTP POST
                     ▼
               FastAPI API
                (main.py)
                     │
                     ▼
           Feature Engineering
                     │
                     ▼
          LightGBM Classifier
                     │
                     ▼
        Irrigation Prediction
          Low / Medium / High
The entire application can run inside a single Docker container.

🛠️ Tech Stack
| Component        | Technology    |
| ---------------- | ------------- |
| Language         | Python        |
| Machine Learning | LightGBM      |
| Data Processing  | Pandas, NumPy |
| API              | FastAPI       |
| Validation       | Pydantic      |
| Frontend         | Streamlit     |
| Containerization | Docker        |
| Server           | Uvicorn       |
| Version Control  | Git / GitHub  |

📁 Project Structure
.
├── main.py
├── irrigation.py
├── irrigation_model.pkl
├── requirements.txt
├── Dockerfile
├── start.sh
└── .dockerignore
File Description

main.py
FastAPI backend responsible for loading the trained model, preparing inference features, and returning predictions.

irrigation.py
Streamlit frontend providing an interactive interface for entering field and crop conditions.

irrigation_model.pkl
Serialized LightGBM model artifact and associated inference metadata.

requirements.txt
Python dependencies required to run the application.

Dockerfile
Build instructions for packaging the complete application into a Docker image.

start.sh
Starts both FastAPI and Streamlit inside the container.

.dockerignore
Prevents unnecessary datasets, notebooks, environments, and temporary files from being included in the Docker build context.

💻 Running Locally
1. Clone the repository
git clone https://github.com/deaminaa/irrigation-need-prediction-using-fastapi.git
cd irrigation-need-prediction-using-fastapi
2. Install dependencies
pip install -r requirements.txt
3. Start the FastAPI backend
uvicorn main:app --reload

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs
4. Start the Streamlit frontend

In another terminal:

streamlit run irrigation.py

The application will normally be available at:

http://localhost:8501
