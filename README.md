# Census Income Prediction

A full-stack machine-learning app that predicts whether a person earns **more than $50K per year** from US Census demographic and employment attributes (the classic *Adult / Census Income* classification task). It includes two trained models, a validated FastAPI backend and a simple web frontend.

Two models are trained and packaged side by side so they can be compared and served interchangeably:

| Model | Algorithm | Artifact | Characteristics |
|---|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` (L2, `max_iter=2000`, `random_state=42`) | `models/census_logistic_model.pkl` | Fast, interpretable coefficients |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` (`random_state=42`) | `models/census_tree_model.pkl` | Non-linear, captures feature interactions |

Each model has a companion `*_columns.pkl` file that stores the exact, ordered list of input features it was trained on. This keeps training and inference aligned and prevents shape-mismatch errors.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Model Input Schema](#model-input-schema)
- [Serving with FastAPI](#serving-with-fastapi)
- [Model Details & Evaluation](#model-details--evaluation)
- [Reproducibility Notes](#reproducibility-notes)
- [Roadmap](#roadmap)

---

## Project Structure

```text
.
├── backend/
│   └── main.py                      # FastAPI backend with Pydantic validation
├── frontend/
│   └── index.html                   # Web UI (HTML + vanilla JavaScript)
├── ml/
│   └── census_training.ipynb        # Notebook used to train and evaluate the models
├── data/
│   └── adult.csv                    # UCI Adult / Census Income dataset
├── models/
│   ├── census_logistic_model.pkl
│   ├── census_logistic_columns.pkl
│   ├── census_tree_model.pkl
│   └── census_tree_columns.pkl
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## Tech Stack

- **Python** 3.9+
- **scikit-learn**: model training and inference
- **pandas**: feature preparation
- **joblib**: model persistence and loading
- **FastAPI** + **Uvicorn**: REST API
- **HTML / JavaScript**: frontend

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/shanzaasad91/adult-income-predictor.git
cd adult-income-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API server
uvicorn backend.main:app --reload
```

Then either:

- open `frontend/index.html` in your browser to use the web app, or
- visit `http://127.0.0.1:8000/docs` for the interactive Swagger documentation.

> **Important:** the models were saved with **scikit-learn 1.6.1**. Using a different version can trigger an `InconsistentVersionWarning` or give wrong results, so keep `scikit-learn==1.6.1` in `requirements.txt`.

## Model Input Schema

The API validates every request with Pydantic (`Literal` and `Field`). Invalid input is rejected with a `422 Validation Error`.

**Numeric features**

| Field | Allowed range |
|---|---|
| `age` | 17 to 90 |
| `hours_per_week` | 1 to 99 |
| `capital_gain` | 0 to 99999 |
| `capital_loss` | 0 to 4356 |

**Categorical features:** `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`. Each accepts only the category values seen in the training data.

**Target:** binary class, `1` = income `>50K`, `0` = income `<=50K`.

## Serving with FastAPI

`backend/main.py` builds a pandas DataFrame from the incoming JSON, one-hot encodes it with `pd.get_dummies()`, and aligns it to the saved `*_columns.pkl` with `reindex(..., fill_value=0)`. The model always receives exactly the features it was trained on, and categories absent from a single request are filled with zeros.

The API returns the predicted class and the probability of earning more than $50K.

**Example request**

```bash
curl -X POST http://127.0.0.1:8000/predict/census-logistic \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "workclass": "Private", "education": "Masters",
       "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
       "relationship": "Husband", "race": "White", "sex": "Male",
       "hours_per_week": 50, "capital_gain": 0, "capital_loss": 0,
       "native_country": "United-States"}'
```

**Example response**

```json
{
  "prediction": ">50K",
  "probability_over_50k": 0.8123
}
```

## Model Details & Evaluation

### Evaluation results

| Model | Accuracy |
|---|---|
| Logistic Regression | 84.25% |
| Decision Tree | 81.67% |

About 76% of records in the dataset earn `<=50K`, so a model that always predicts `<=50K` would already score about 76% accuracy. Both models beat that baseline, but accuracy alone hides how well each one finds the `>50K` class.
### Preprocessing

- Rows with missing values (`?`) are removed.
- `education-num` and `fnlwgt` are dropped to avoid redundancy and noise.
- Categorical features are one-hot encoded with `pd.get_dummies()`.
- No feature scaling is applied.

### Limitations

- **Decision tree capacity:** the saved tree is not depth-limited, so it likely overfits. Treat its probabilities as coarse.
- **Single metric:** only accuracy is reported (see above).
- **Fairness and responsible use:** features include `sex` and `race`, and the underlying 1994 data reflects historical societal bias. **Do not use this model for real decisions about individuals** (hiring, lending, housing, and so on). It is intended strictly for learning and demonstration.
- **Data vintage:** predictions reflect 1990s wages and do not account for inflation.
- **Pickle safety:** `.pkl` files can execute arbitrary code when loaded. Only load model files from sources you trust.

## Reproducibility Notes

- Both models use `random_state=42` and were trained with `scikit-learn==1.6.1`.
- The raw dataset is in `data/adult.csv`, and the full training pipeline is in `ml/census_training.ipynb`.
- If you retrain, regenerate both the model files and their matching `*_columns.pkl` files so they stay in sync.


