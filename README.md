# Aviation Fuel Uplift Prediction

![Model comparison](model_comparison.jpg)

Machine-learning models that predict how much jet fuel (in litres) an aircraft uplifts at departure, using the route, the aircraft type and the landed fuel price at both ends of the sector.

Four models are compared against a simple "predict the average" benchmark:

**Linear Regression → Random Forest → XGBoost → CatBoost**

> **Note on data:** the dataset is **simulated** to mirror the structure of real flight-wise uplift records, and it is not included in this repository. The results show that the pipeline works end to end. They are not a claim about real-world accuracy, since real uplift data will be noisier.

---

## Why this problem

Airlines don't refuel to the same level on every flight. How much they uplift depends on the sector, the aircraft and, potentially, fuel price differences between airports. Buying extra fuel where it is cheaper is known as *tankering*.

A good uplift forecast helps fuel suppliers and airport operators plan stock, hydrant capacity and supply logistics.

**Uplift is not the same as fuel burn.** Uplift also depends on fuel already on board and on commercial decisions, so the two quantities are not treated as interchangeable here.

---

## Data

Three separate files, split chronologically so models are always judged on a later period than they learned from:

| File | Flights | Purpose |
|---|---|---|
| `training.xlsx` | 15,385 | Fit the models |
| `validation.xlsx` | 3,701 | Compare models and tune hyperparameters |
| `test.xlsx` | — | Held back for one final evaluation |

**Inputs**

| Column | Type |
|---|---|
| Day of Week | Categorical |
| Aircraft | Categorical |
| Turboprop | Categorical (Yes/No) |
| To IATA (destination) | Categorical |
| From Landed Price (Rs/KL) | Numeric |
| To Landed Price (Rs/KL) | Numeric |

**Target:** `Qty Uplifted (L)`

---

## Method

All models use scikit-learn `Pipeline`s, so preprocessing is learned from the training data only and applied identically at prediction time.

- **Cleaning:** consistent text formatting, numeric parsing, and validation of weekday and Yes/No fields. Invalid prices become missing values.
- **Missing values:** the median for prices and an `"Unknown"` category for text fields, fitted on training data only.
- **Encoding:** one-hot encoding for linear regression, the random forest and XGBoost. CatBoost reads the categories directly as text.
- **Tuning:** each tree model is tuned on validation MAE, with a fixed random seed.

| Model | Search grid | Best setting |
|---|---|---|
| Random forest | 500 trees × `min_samples_leaf` {1, 3, 5, 10} × `max_features` {1.0, 0.5, sqrt} | leaf 5, features 0.5 |
| XGBoost | `n_estimators` {300, 600} × `max_depth` {4, 6} × `learning_rate` {0.05, 0.1} | 300 trees, depth 4, lr 0.1 |
| CatBoost | `iterations` {500, 1000} × `depth` {4, 6} × `learning_rate` {0.05, 0.1} | 500 trees, depth 4, lr 0.05 |

- **Explainability:** permutation importance on validation data. Each input is shuffled in turn to measure how much the error worsens.

---

## Results (validation set)

| Model | MAE (L) | RMSE (L) | R² | MAE vs linear |
|---|---|---|---|---|
| Benchmark (training mean) | 2,997 | — | — | — |
| Linear regression | 473 | 582 | 0.966 | — |
| CatBoost | 365 | 490 | 0.976 | −22.8% |
| Random forest | 353 | 469 | 0.978 | −25.5% |
| **XGBoost** | **345** | **462** | **0.979** | **−27.2%** |

> Test-set results will be added after a single final run of all models.

### Key findings

1. **Moving to tree ensembles is the big win.** Every tree-based model cuts error by 23–27% compared with linear regression. Linear regression assumes a route adds the same litres whatever aircraft flies it, while tree models learn these interactions.
2. **The choice of ensemble matters much less.** XGBoost beats the random forest by about 8 L per flight (around 2%), and both were tuned on the same validation set, so they are effectively tied. The gain comes from capturing interactions, not from the specific algorithm.
3. **CatBoost came last of the ensembles.** Some of its settings overfit badly (validation MAE around 590 L at depth 6 with learning rate 0.1), and it leaned more heavily on fuel prices. One likely reason is that prices drift over time and tree models can't extrapolate beyond the training range. This is a hypothesis, not a proven cause.

### What drives the prediction

These patterns are consistent across all tree models:

- **Destination (To IATA) is always the dominant driver.** It effectively stands in for sector distance.
- **The aircraft effect comes next.** Aircraft and Turboprop carry overlapping information, so models split the credit between them differently. Read the two together.
- **Fuel price differences add little,** which points to a weak tankering signal in this data. This is an association found by predictive models, not evidence that prices *cause* changes in uplift.
- **Day of week barely matters.**

---

## How to run

```bash
pip install -r requirements.txt

python Regression.py      # linear regression   -> outputs/
python RandomForest.py    # random forest       -> outputs_rf/
python Boosting.py        # XGBoost + CatBoost  -> outputs_boost/ (plus a table of all models)

# Final, one-time evaluation on the held-out test set (run Boosting.py last)
python Regression.py --evaluate-test
python RandomForest.py --evaluate-test
python Boosting.py --evaluate-test
```

Keep `training.xlsx`, `validation.xlsx` and `test.xlsx` in the same folder as the scripts. In Spyder, run each file with F5, or add `--evaluate-test` after `--wdir` in the `%runfile` command.

**Outputs**

| File | Contents |
|---|---|
| `*_predictions*.xlsx` | Every flight with its predicted uplift and error |
| `evaluation_metrics*.csv` | MAE, RMSE and R² per model and dataset |
| `tuning_*.csv` | Validation MAE for every hyperparameter setting |
| `feature_importance_*.csv` | Permutation importance per model |
| `all_models_comparison.csv` | All four models side by side |
| `*.joblib` | Saved models with their preprocessing |

### Predicting a new flight

```python
import joblib
import pandas as pd

model = joblib.load("outputs_rf/rf_fuel_uplift_model.joblib")

flight = pd.DataFrame([{
    "Day of Week": "Monday",
    "Aircraft": "320",
    "Turboprop": "No",
    "To IATA": "DEL",
    "From Landed Price (Rs/KL)": 91974,   # illustrative values
    "To Landed Price (Rs/KL)": 85358,
}])

print(model.predict(flight))   # ≈ 9,078 L on the simulated data
```

The XGBoost model loads the same way. The CatBoost pipeline uses a helper function defined in `Boosting.py`, so load it from inside that script or import the function first.

---

## Limitations and next steps

- **Simulated data.** Real records would add noise from weather, payload and operational decisions.
- **Missing inputs.** Fuel on board before refuelling, payload and sector distance aren't included. Any of these could improve accuracy, provided they are known at prediction time.
- **Unfamiliar routes or aircraft.** Predictions for categories absent from training are unreliable, and the scripts flag them.
- **Planned additions:** a final test-set evaluation, actual-vs-predicted and residual plots, error broken down by sector and aircraft, and time-aware cross-validation for tuning.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · CatBoost · joblib · openpyxl

## Author

**Kaustav Palit**. EPGP, IIM Bangalore, with about 9 years in aviation fuel operations at IOCL and BPCL.
