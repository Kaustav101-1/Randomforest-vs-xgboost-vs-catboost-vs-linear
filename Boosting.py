# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:27:59 2026

@author: Kaustav palit
"""

# -*- coding: utf-8 -*-
"""
Predict fuel uplift using gradient boosting (XGBoost and CatBoost),
and compare against the linear regression and random forest results.

Standalone: uses the same Excel files, columns and cleaning as
Regression.py and RandomForest.py. Keep it in the same folder.

Install once:
    pip install xgboost catboost

Run (validation only - use this while choosing the final model):
    python fuel_uplift_boosting.py

Final, one-time test evaluation (only after the final model is chosen):
    python fuel_uplift_boosting.py --evaluate-test

Outputs go to outputs_boost/. If the linear (outputs/) and random
forest (outputs_rf/) metric files exist, a combined comparison table
of all models is printed and saved.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Running line-by-line in a console (e.g. a Spyder cell).
    BASE_DIR = Path.cwd()

TRAIN_FILE = BASE_DIR / "training.xlsx"
VALIDATION_FILE = BASE_DIR / "validation.xlsx"
TEST_FILE = BASE_DIR / "test.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs_boost"
LINEAR_METRICS = BASE_DIR / "outputs" / "evaluation_metrics.csv"
RF_METRICS = BASE_DIR / "outputs_rf" / "evaluation_metrics_rf.csv"

CATEGORICAL_FEATURES = [
    "Day of Week",
    "Aircraft",
    "Turboprop",
    "To IATA",
]

NUMERIC_FEATURES = [
    "From Landed Price (Rs/KL)",
    "To Landed Price (Rs/KL)",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET = "Qty Uplifted (L)"

VALID_WEEKDAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}

RANDOM_STATE = 42

# Small grids, 8 settings per model, scored on validation MAE.
#   n_estimators / iterations : number of trees built in sequence
#   max_depth / depth         : how complex each tree can be
#   learning_rate             : how much each new tree corrects the last
XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (300, 600)
    for d in (4, 6)
    for lr in (0.05, 0.1)
]

CAT_GRID = [
    {"iterations": n, "depth": d, "learning_rate": lr}
    for n in (500, 1000)
    for d in (4, 6)
    for lr in (0.05, 0.1)
]


# ---------------------------------------------------------
# 2. Cleaning and loading (same rules as the other scripts)
# ---------------------------------------------------------

def clean_features(data):
    """Standardize formatting without learning from the data."""
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()

    missing_columns = [
        column for column in FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing input columns: {missing_columns}")

    for column in CATEGORICAL_FEATURES:
        values = data[column].astype("string").str.strip()

        if column == "Aircraft":
            # Excel can read aircraft code 320 as 320.0.
            values = values.str.replace(
                r"\.0$", "", regex=True
            ).str.upper()
        elif column == "To IATA":
            values = values.str.upper()
        else:
            values = values.str.title()

        values = values.replace("", pd.NA)

        data[column] = values.astype(object).where(
            values.notna(), np.nan
        )

    invalid_weekdays = (
        set(data["Day of Week"].dropna()) - VALID_WEEKDAYS
    )

    if invalid_weekdays:
        raise ValueError(
            "Use full weekday names, such as Monday. "
            f"Invalid values: {sorted(invalid_weekdays)}"
        )

    invalid_turboprop = (
        set(data["Turboprop"].dropna()) - {"Yes", "No"}
    )

    if invalid_turboprop:
        raise ValueError(
            "Turboprop must contain Yes or No. "
            f"Invalid values: {sorted(invalid_turboprop)}"
        )

    for column in NUMERIC_FEATURES:
        values = (
            data[column]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        numbers = pd.to_numeric(
            values, errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)

        # Invalid prices become missing and are later imputed.
        numbers[~np.isfinite(numbers) | (numbers <= 0)] = np.nan

        data[column] = numbers

    return data


def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_excel(file_path)
    data.columns = data.columns.astype(str).str.strip()

    if data.columns.duplicated().any():
        raise ValueError(f"{file_path.name}: duplicate column headings.")

    if TARGET not in data.columns:
        raise ValueError(
            f"{file_path.name}: missing target column '{TARGET}'."
        )

    data = clean_features(data)

    target_values = (
        data[TARGET]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    data[TARGET] = pd.to_numeric(
        target_values, errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)

    invalid_target = ~np.isfinite(data[TARGET]) | (data[TARGET] < 0)

    if invalid_target.any():
        excel_rows = np.flatnonzero(invalid_target.to_numpy()) + 2
        raise ValueError(
            f"{file_path.name}: invalid uplift in "
            f"{invalid_target.sum()} rows. "
            f"Example Excel rows: {excel_rows[:10].tolist()}"
        )

    if data.empty:
        raise ValueError(f"{file_path.name}: dataset is empty.")

    print(f"\nLoaded {file_path.name}: {len(data):,} flights")

    missing_counts = data[FEATURES].isna().sum()

    if missing_counts.any():
        print("Missing inputs to be imputed:")
        print(missing_counts[missing_counts > 0].to_string())

    return data


def report_unfamiliar_categories(training_inputs, new_inputs):
    for column in CATEGORICAL_FEATURES:
        known = set(training_inputs[column].dropna())
        incoming = set(new_inputs[column].dropna())
        unfamiliar = incoming - known

        if unfamiliar:
            print(
                f"Unfamiliar categories in {column}: "
                f"{sorted(unfamiliar)}"
            )


# ---------------------------------------------------------
# 3. Build the two boosting pipelines
# ---------------------------------------------------------

def build_xgboost(n_estimators, max_depth, learning_rate):
    """XGBoost with one-hot encoded categories (same as the forest)."""
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="Unknown"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessing = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC_FEATURES,
        ),
    ])

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,          # each tree sees 80% of flights
        colsample_bytree=0.8,   # and 80% of input columns
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", model),
    ])


def prepare_for_catboost(X):
    """
    CatBoost reads text categories directly, so no one-hot encoding.
    Missing categories become 'Unknown'; missing prices stay NaN,
    which CatBoost handles itself. Learns nothing from the data.
    """
    X = X[FEATURES].copy()

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("Unknown").astype(str)

    return X


def build_catboost(iterations, depth, learning_rate):
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        cat_features=CATEGORICAL_FEATURES,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )

    return Pipeline([
        ("preprocessing", FunctionTransformer(prepare_for_catboost)),
        ("regression", model),
    ])


# ---------------------------------------------------------
# 4. Tune each model on validation (fit on training only)
# ---------------------------------------------------------

def tune(name, builder, grid, X_train, y_train, validation):
    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    rows = []
    best_model = None
    best_params = None
    best_mae = np.inf

    print(f"\nTuning {name} on validation set ({len(grid)} settings)...")

    for params in grid:
        model = builder(**params)
        model.fit(X_train, y_train)

        mae = mean_absolute_error(y_val, model.predict(X_val))
        rows.append({**params, "Validation_MAE_L": mae})

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_params = params

    tuning = (
        pd.DataFrame(rows)
        .sort_values("Validation_MAE_L")
        .reset_index(drop=True)
    )

    print(tuning.round(3).to_string(index=False))

    tuning.to_csv(
        OUTPUT_DIR / f"tuning_{name.lower()}.csv",
        index=False,
    )

    print(f"Best {name} settings: {best_params}")

    return best_model, best_params


# ---------------------------------------------------------
# 5. Evaluate and save predictions
# ---------------------------------------------------------

def evaluate_model(model, data, dataset_name, model_name, training_mean):
    X = data[FEATURES]
    y = data[TARGET]

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = (
        r2_score(y, predictions)
        if len(y) > 1 and y.nunique() > 1
        else np.nan
    )

    benchmark_mae = mean_absolute_error(
        y, np.full(len(y), training_mean)
    )

    # Unlike a random forest, boosting can predict below zero.
    negative = int((predictions < 0).sum())

    metrics = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Flights": len(data),
        "MAE_L": mae,
        "RMSE_L": rmse,
        "R2": r2,
        "Training_mean_benchmark_MAE_L": benchmark_mae,
        "Negative_predictions": negative,
    }

    print(f"\n{dataset_name} performance ({model_name})")
    print(f"MAE:  {mae:,.1f} L")
    print(f"RMSE: {rmse:,.1f} L")
    print(f"R²:   {r2:.3f}")
    print(f"Negative predictions: {negative}")

    results = data.copy()
    results["Predicted Uplift (L)"] = predictions
    results["Error (L)"] = results["Predicted Uplift (L)"] - results[TARGET]
    results["Absolute Error (L)"] = results["Error (L)"].abs()

    prediction_file = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}_predictions_{model_name.lower()}.xlsx"
    )

    results.to_excel(prediction_file, index=False)
    print(f"Predictions saved: {prediction_file}")

    return metrics


# ---------------------------------------------------------
# 6. Feature importance for each model
# ---------------------------------------------------------

def report_importance(model, data, model_name):
    """Shuffle one input at a time; measure how much MAE worsens."""
    result = permutation_importance(
        model,
        data[FEATURES],
        data[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "MAE_increase_L": result.importances_mean,
        "Std_L": result.importances_std,
    }).sort_values("MAE_increase_L", ascending=False)

    print(f"\nFeature importance - {model_name} (MAE increase when shuffled)")
    print(importance.round(1).to_string(index=False))

    importance.to_csv(
        OUTPUT_DIR / f"feature_importance_{model_name.lower()}.csv",
        index=False,
    )


# ---------------------------------------------------------
# 7. One table for all four models
# ---------------------------------------------------------

def compare_all_models(boost_metrics):
    frames = []

    if LINEAR_METRICS.exists():
        linear = pd.read_csv(LINEAR_METRICS)
        linear["Model"] = "Linear regression"
        frames.append(linear)
    else:
        print("\nLinear results not found - run Regression.py first.")

    if RF_METRICS.exists():
        rf = pd.read_csv(RF_METRICS)
        rf["Model"] = "Random forest"
        frames.append(rf)
    else:
        print("Random forest results not found - run RandomForest.py first.")

    frames.append(pd.DataFrame(boost_metrics))

    columns = ["Model", "Dataset", "MAE_L", "RMSE_L", "R2"]
    combined = pd.concat([f[columns] for f in frames], ignore_index=True)

    for dataset, group in combined.groupby("Dataset", sort=False):
        table = group.sort_values("MAE_L").reset_index(drop=True)

        if "Linear regression" in set(table["Model"]):
            linear_mae = table.loc[
                table["Model"] == "Linear regression", "MAE_L"
            ].iloc[0]
            table["MAE_vs_linear_%"] = (
                table["MAE_L"] / linear_mae - 1
            ) * 100

        print(f"\nAll models - {dataset} (best first)")
        print(table.round(3).to_string(index=False))

    combined.to_csv(OUTPUT_DIR / "all_models_comparison.csv", index=False)


# ---------------------------------------------------------
# 8. Main workflow
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost and CatBoost fuel uplift models."
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the finalized models on test.xlsx.",
    )

    # parse_known_args ignores extra arguments Spyder may pass.
    args, _ = parser.parse_known_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = load_dataset(TRAIN_FILE)
    validation = load_dataset(VALIDATION_FILE)

    X_train = training[FEATURES]
    y_train = training[TARGET]

    empty_features = [
        column for column in FEATURES
        if X_train[column].isna().all()
    ]

    if empty_features:
        raise ValueError(
            "These inputs have no usable training values: "
            f"{empty_features}"
        )

    report_unfamiliar_categories(X_train, validation[FEATURES])

    training_mean = float(y_train.mean())

    test = None
    if args.evaluate_test:
        test = load_dataset(TEST_FILE)
        report_unfamiliar_categories(X_train, test[FEATURES])

    candidates = [
        ("XGBoost", build_xgboost, XGB_GRID),
        ("CatBoost", build_catboost, CAT_GRID),
    ]

    all_metrics = []

    for name, builder, grid in candidates:
        model, best_params = tune(
            name, builder, grid, X_train, y_train, validation
        )

        all_metrics.append(
            evaluate_model(model, validation, "Validation", name, training_mean)
        )

        if test is not None:
            all_metrics.append(
                evaluate_model(model, test, "Test", name, training_mean)
            )

        report_importance(model, validation, name)

        model_path = OUTPUT_DIR / f"{name.lower()}_fuel_uplift_model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

        pd.DataFrame([best_params]).to_csv(
            OUTPUT_DIR / f"best_params_{name.lower()}.csv",
            index=False,
        )

    pd.DataFrame(all_metrics).to_csv(
        OUTPUT_DIR / "evaluation_metrics_boost.csv",
        index=False,
    )

    compare_all_models(all_metrics)

    # Same illustrative flight as the other scripts.
    new_flight = clean_features(pd.DataFrame([{
        "Day of Week": "Monday",
        "Aircraft": "320",
        "Turboprop": "No",
        "To IATA": "DEL",
        "From Landed Price (Rs/KL)": 91974,
        "To Landed Price (Rs/KL)": 85358,
    }]))

    report_unfamiliar_categories(X_train, new_flight[FEATURES])

    print("\nIllustrative flight predicted uplift:")
    for name, _, _ in candidates:
        loaded = joblib.load(
            OUTPUT_DIR / f"{name.lower()}_fuel_uplift_model.joblib"
        )
        value = loaded.predict(new_flight[FEATURES])[0]
        print(f"  {name}: {value:,.0f} L")


if __name__ == "__main__":
    main()