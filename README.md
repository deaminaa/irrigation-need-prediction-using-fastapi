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


Low
Medium
High

The saved model artifact contains the trained model together with the feature configuration required for inference.


#### Feature Engineering

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
🐳 Run with Docker

The application is packaged as a single Docker image containing both the FastAPI backend and Streamlit frontend.

Pull the image
docker pull mueezuddin/irrigation-predictor:latest
Run the container
docker run --name irrigation-app \
  -p 8501:8501 \
  -p 8000:8000 \
  mueezuddin/irrigation-predictor:latest

Then open:

http://localhost:8501

FastAPI documentation:

http://localhost:8000/docs
🔌 API
GET /

Basic application status endpoint.

GET /health

Health-check endpoint confirming that the API and model are loaded.

POST /predict

Accepts field and crop information and returns the predicted irrigation requirement together with class probabilities.

Example request:

{
  "Soil_Type": "Sandy",
  "Soil_pH": 6.5,
  "Soil_Moisture": 35,
  "Electrical_Conductivity": 0.8,
  "Temperature_C": 28,
  "Humidity": 60,
  "Rainfall_mm": 5,
  "Wind_Speed_kmh": 12,
  "Crop_Type": "Wheat",
  "Crop_Growth_Stage": "Vegetative",
  "Season": "Rabi",
  "Irrigation_Type": "Drip",
  "Water_Source": "Groundwater",
  "Mulching_Used": "Yes",
  "Previous_Irrigation_mm": 10
}

Example response structure:

{
  "prediction": "Low",
  "class_id": 0,
  "probabilities": {
    "Low": 0.97,
    "Medium": 0.03,
    "High": 0.00
  }
}
📊 Model Output

The application returns:

Predicted irrigation requirement
Numeric class ID
Probability for each irrigation class

This makes the system useful not only for classification but also for understanding model confidence.

🎯 Key Engineering Highlights

This project demonstrates an end-to-end ML deployment workflow rather than only model training:

Multiclass machine learning with LightGBM
Reusable feature-engineering pipeline
Model serialization and inference
REST API development with FastAPI
Interactive ML application with Streamlit
Input validation using Pydantic
Dockerized deployment
Single-container application architecture
API documentation through Swagger/OpenAPI
Reproducible deployment through Docker Hub
📦 Docker Image

Docker image:

mueezuddin/irrigation-predictor:latest

Docker Hub:

https://hub.docker.com/r/mueezuddin/irrigation-predictor

🔮 Future Improvements

Possible future extensions include:

Deploying the container to AWS
Adding model monitoring and logging
Automated CI/CD with GitHub Actions
Adding prediction history and analytics
Improving model interpretability with SHAP
Adding authentication and production API security
👨‍💻 Author

Mueezuddin Ahmed

GitHub:
https://github.com/deaminaa

📄 License

This project is licensed under the MIT License.


### One thing I'd change before you commit

Your repository URL in the screenshot is:

```text
github.com/deaminaa/irrigation-need-prediction-using-fastapi

so I deliberately used that exact URL in the README.

Also, I would not add fake accuracy numbers, Kaggle rankings, deployment claims, or business impact to this README unless we have the actual evidence in the project. The current version sells the engineering work—which is genuinely strong—without overstating anything.

Paste this into README.md, then don't commit it yet. Send me a screenshot of the README editor and I'll quickly check the formatting before you hit commit.
