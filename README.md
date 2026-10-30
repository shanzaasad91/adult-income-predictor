# Census Income Prediction

Machine-learning service that predicts whether a person earns **more than $50K per year** from US Census demographic and employment attributes (the classic *Adult / Census Income* classification task).

Two models are trained and packaged side by side so they can be compared and served interchangeably:

| Model | Algorithm | Artifact | Characteristics |
|---|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` (L2, `max_iter=1000`, `random_state=42`) | `census_logistic_model.pkl` | Fast, small (~4 KB), interpretable coefficients |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` (`random_state=42`, unconstrained depth) | `census_tree_model.pkl` | Non-linear, captures feature interactions, larger (~700 KB) |

Each model ships with a companion `*_columns.pkl` file that stores the exact, ordered list of 79 input features it was trained on. This guarantees training/inference feature alignment.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Model Input Schema](#model-input-schema)
- [Using the Models](#using-the-models)
- [Serving with FastAPI](#serving-with-fastapi)
- [Model Details & Limitations](#model-details--limitations)
- [Reproducibility Notes](#reproducibility-notes)
- [Security Note](#security-note)
- [Roadmap](#roadmap)

---

## Project Structure

```
.
├── census_logistic_model.pkl     # Trained Logistic Regression model
├── census_logistic_columns.pkl   # Ordered feature list (79) for the logistic model
├── census_tree_model.pkl         # Trained Decision Tree model
├── census_tree_columns.pkl       # Ordered feature list (79) for the tree model
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignores venv/, __pycache__/, *.pyc, .ipynb_checkpoints/
└── README.md
```

> **Note:** the repository currently ships the trained artifacts only. The API entry point (e.g. `main.py`) and training notebook are expected to live alongside them; see [Serving with FastAPI](#serving-with-fastapi).

## Tech Stack

- **Python** 3.9+
- **scikit-learn**: model training and inference
- **pandas**: feature preparation
- **joblib**: model persistence / loading
- **FastAPI** + **Uvicorn**: REST API serving layer

## Getting Started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Important:** the model files were serialized with **scikit-learn 1.6.1**. Loading them with a different version emits an `InconsistentVersionWarning` and may produce incorrect results. For reliable behavior, pin the version:
>
> ```bash
> pip install "scikit-learn==1.6.1"
> ```
>
> and set `scikit-learn==1.6.1` in `requirements.txt`.

## Model Input Schema

Both models expect a single-row (or multi-row) `DataFrame` with **79 columns**, in the order stored in the `*_columns.pkl` files.

**Numeric features (4):** `age`, `capital-gain`, `capital-loss`, `hours-per-week`

**One-hot encoded categorical features (75):** created from the original categorical fields below with `drop_first=True`. The dropped *reference* category for each field is shown, so a record in that category has all of that field's dummy columns set to `0`.

| Source field | Encoded as | Reference category (all zeros) |
|---|---|---|
| `workclass` | `workclass_<value>` | `Federal-gov` |
| `marital-status` | `marital-status_<value>` | `Divorced` |
| `occupation` | `occupation_<value>` | `Adm-clerical` |
| `relationship` | `relationship_<value>` | `Husband` |
| `race` | `race_<value>` | `Amer-Indian-Eskimo` |
| `sex` | `sex_Male` | `Female` |
| `native-country` | `native-country_<value>` | `Cambodia` |

**Target:** binary class, `1` = income `>50K`, `0` = income `<=50K`.

## Using the Models

```python
import joblib
import pandas as pd

# Load the model and its expected feature order
model   = joblib.load("census_logistic_model.pkl")   # or census_tree_model.pkl
columns = joblib.load("census_logistic_columns.pkl") # or census_tree_columns.pkl

# Build one record: start from all zeros, then set the active features
record = dict.fromkeys(columns, 0)
record.update({
    "age": 39,
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "workclass_State-gov": 1,
    "marital-status_Never-married": 1,
    "occupation_Prof-specialty": 1,
    "relationship_Not-in-family": 1,
    "race_White": 1,
    "sex_Male": 1,
    "native-country_United-States": 1,
})

X = pd.DataFrame([record], columns=columns)   # enforces column order

prediction  = int(model.predict(X)[0])              # 0 or 1
probability = float(model.predict_proba(X)[0][1])   # P(income > 50K)

print("Income >50K" if prediction else "Income <=50K", f"(p={probability:.2f})")
```

To score raw, human-readable records (e.g. `{"workclass": "Private", ...}`), one-hot encode them and align to the saved columns:

```python
raw = pd.DataFrame([{"age": 39, "workclass": "State-gov", "sex": "Male", ...}])
X = pd.get_dummies(raw).reindex(columns=columns, fill_value=0)
```

`reindex(..., fill_value=0)` guarantees the correct column order and silently handles categories absent from a single request.

## Serving with FastAPI

A minimal service layout (adapt to your `main.py`):

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="Census Income Prediction API")

MODELS = {
    "logistic": (joblib.load("census_logistic_model.pkl"), joblib.load("census_logistic_columns.pkl")),
    "tree":     (joblib.load("census_tree_model.pkl"),     joblib.load("census_tree_columns.pkl")),
}

class Person(BaseModel):
    age: int
    workclass: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40

@app.post("/predict/{model_name}")
def predict(model_name: str, person: Person):
    model, columns = MODELS[model_name]
    raw = pd.DataFrame([{
        "age": person.age,
        "capital-gain": person.capital_gain,
        "capital-loss": person.capital_loss,
        "hours-per-week": person.hours_per_week,
        "workclass": person.workclass,
        "marital-status": person.marital_status,
        "occupation": person.occupation,
        "relationship": person.relationship,
        "race": person.race,
        "sex": person.sex,
        "native-country": person.native_country,
    }])
    X = pd.get_dummies(raw).reindex(columns=columns, fill_value=0)
    return {
        "model": model_name,
        "prediction": int(model.predict(X)[0]),
        "probability_over_50k": float(model.predict_proba(X)[0][1]),
    }
```

Run it:

```bash
uvicorn main:app --reload
```

Then open the interactive docs at **http://127.0.0.1:8000/docs**.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict/logistic \
  -H "Content-Type: application/json" \
  -d '{"age": 39, "workclass": "State-gov", "marital_status": "Never-married",
       "occupation": "Prof-specialty", "relationship": "Not-in-family",
       "race": "White", "sex": "Male", "native_country": "United-States",
       "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40}'
```

## Model Details & Limitations

- **Unscaled features:** the numeric inputs are used as-is (no scaler is saved). Do not apply scaling at inference time.
- **Decision Tree capacity:** the saved tree is fully grown (depth 50, ~4,400 leaves), which typically **overfits**. Treat its probabilities as coarse; limiting `max_depth` / `min_samples_leaf` and re-training is recommended.
- **Dropped fields:** `education`, `education-num`, and `fnlwgt` are not part of the model's feature set.
- **Fairness & responsible use:** features include `sex`, `race`, and `native-country`, and the underlying data reflects historical societal bias. **Do not use this model for real decisions about individuals** (hiring, lending, housing, etc.). It is intended for learning, experimentation, and demonstration.
- **Data vintage:** census data from the 1990s; predictions do not reflect current wages or inflation.

### Evaluation results

Add your held-out test results here once finalized:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Decision Tree | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Reproducibility Notes

- Both models use `random_state=42`.
- Pin dependency versions (especially `scikit-learn==1.6.1`) so that unpickling is deterministic.
- If you retrain, regenerate **both** the model and its `*_columns.pkl` together; they must stay in sync.
- The ignore file is currently uploaded as `_gitignore`; rename it to `.gitignore` so Git picks it up.

## Security Note

`.pkl` files can execute arbitrary code when loaded. Only load model files from sources you trust, and never accept pickled files from end users.

## Roadmap

- [ ] Pin exact dependency versions in `requirements.txt`
- [ ] Add training notebook / script for full reproducibility
- [ ] Publish evaluation metrics and confusion matrices
- [ ] Regularize the decision tree (depth / leaf-size limits) or add an ensemble (Random Forest, Gradient Boosting)
- [ ] Add input validation against the allowed category values
- [ ] Add `/health` endpoint, tests, and a Dockerfile

## License


