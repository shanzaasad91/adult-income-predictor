from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import os

# Resolve paths from the project root so it works from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Adult Income Predictor",
    version="0.2.0"
)

# Allow the browser (file:// or localhost) to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

# ---------- Load Models (UPDATED to look inside the 'models/' folder) ----------
logistic_model   = joblib.load(os.path.join(BASE_DIR, "models", "census_logistic_model.pkl"))
tree_model       = joblib.load(os.path.join(BASE_DIR, "models", "census_tree_model.pkl"))
logistic_columns = joblib.load(os.path.join(BASE_DIR, "models", "census_logistic_columns.pkl"))
tree_columns     = joblib.load(os.path.join(BASE_DIR, "models", "census_tree_columns.pkl"))


# ---------- Input Schema with Validation ----------
class CensusData(BaseModel):
    age: int = Field(..., ge=17, le=90, description="Age in years")
    workclass: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc", 
        "Federal-gov", "Local-gov", "State-gov", "Without-pay"
    ]
    education: Literal[
        "Bachelors", "HS-grad", "Masters", "Some-college", 
        "Assoc-voc", "11th", "Doctorate", "Prof-school"
    ]
    marital_status: Literal[
        "Married-civ-spouse", "Never-married", "Divorced", 
        "Separated", "Widowed"
    ]
    occupation: Literal[
        "Exec-managerial", "Prof-specialty", "Craft-repair", 
        "Adm-clerical", "Sales", "Other-service"
    ]
    relationship: Literal[
        "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"
    ]
    race: Literal[
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ]
    sex: Literal["Male", "Female"]
    hours_per_week: int = Field(..., ge=1, le=99, description="Hours worked per week")
    capital_gain: int = Field(..., ge=0, le=99999)
    capital_loss: int = Field(..., ge=0, le=4356)
    native_country: Literal[
        "United-States", "Mexico", "Philippines", "Germany", "Canada", "India"
    ]


# ---------- Helper ----------
def build_input_df(data: CensusData) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame with original dataset column names."""
    values = {
        "age": data.age,
        "workclass": data.workclass,
        "education-num": 0,   # Placeholder, dropped immediately below
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country,
    }
    df = pd.DataFrame([values])
    # Drop 'education-num' placeholder before encoding
    df = df.drop(columns=["education-num"])
    return df


# ---------- Endpoints ----------
@app.post("/predict/census-logistic")
def predict_logistic(data: CensusData):
    input_df = build_input_df(data)
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=logistic_columns, fill_value=0)

    # Get prediction (0 or 1) and probabilities
    prediction = int(logistic_model.predict(input_df)[0])
    probabilities = logistic_model.predict_proba(input_df)[0]
    
    # Round the probability of class 1 (>50K) to 4 decimal places
    prob_over_50k = round(float(probabilities[1]), 4)

    return {
        "prediction": ">50K" if prediction == 1 else "<=50K",
        "probability_over_50k": prob_over_50k
    }


@app.post("/predict/census-tree")
def predict_tree(data: CensusData):
    input_df = build_input_df(data)
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=tree_columns, fill_value=0)

    # Get prediction (0 or 1) and probabilities
    prediction = int(tree_model.predict(input_df)[0])
    probabilities = tree_model.predict_proba(input_df)[0]
    
    # Round the probability of class 1 (>50K) to 4 decimal places
    prob_over_50k = round(float(probabilities[1]), 4)

    return {
        "prediction": ">50K" if prediction == 1 else "<=50K",
        "probability_over_50k": prob_over_50k
    }