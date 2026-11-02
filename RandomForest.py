# -*- coding: utf-8 -*-
"""
Predict fuel uplift using a random forest.

Uses the same Excel files, columns and cleaning as fuel_uplift_linear.py.
Standalone: the cleaning and loading code is copied in, so it does
not need to import fuel_uplift_linear.py. Keep the three Excel files
in the same folder as this script.

Run:
    python fuel_uplift_rf.py

Final test evaluation:
    python fuel_uplift_rf.py --evaluate-test

Outputs go to outputs_rf/ so the linear model's outputs are not overwritten.
If outputs/evaluation_metrics.csv from the linear run exists, a
side-by-side comparison is printed and saved.
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pathlib import Path


# ---------------------------------------------------------
# 1. Configuration (same as the linear script)
# ---------------------------------------------------------

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Running line-by-line in a console (e.g. Spyder cell).
    BASE_DIR = Path.cwd()

TRAIN_FILE = BASE_DIR / "training.xlsx"
VALIDATION_FILE = BASE_DIR / "validation.xlsx"
TEST_FILE = BASE_DIR / "test.xlsx"

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

OUTPUT_DIR = BASE_DIR / "outputs_rf"
LINEAR_METRICS = BASE_DIR / "outputs" / "evaluation_metrics.csv"

RANDOM_STATE = 42
N_TREES = 500

# Small grid, scored on the validation set.
# min_samples_leaf: larger = smoother, less overfitting.
# max_features: share of inputs each split may consider.
PARAM_GRID = [
    {"min_samples_leaf": leaf, "max_features": features}
    for leaf in (1, 3, 5, 10)
    for features in (1.0, 0.5, "sqrt")
]


# ---------------------------------------------------------
# 1a. Shared cleaning and loading (copied from the linear script)
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
# 2. Build the random forest pipeline
# ---------------------------------------------------------

def build_model(min_samples_leaf=1, max_features=1.0):
    # Trees don't need scaling or a dropped dummy column.
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    preprocessing = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
    ])

    forest = RandomForestRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", forest),
    ])


# ---------------------------------------------------------
# 3. Tune on validation (fit on training only)
# ---------------------------------------------------------

def tune_model(X_train, y_train, validation):
    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    rows = []
    best_model = None
    best_mae = np.inf

    print("\nTuning random forest on validation set...")

    for params in PARAM_GRID:
        model = build_model(**params)
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

    print(tuning.round(1).to_string(index=False))

    tuning.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    print(f"\nBest settings: {best_params}")

    return best_model, best_params


# ---------------------------------------------------------
# 4. Evaluate and save predictions
# ---------------------------------------------------------

def evaluate_model(model, data, dataset_name, training_mean):
    X = data[FEATURES]
    y = data[TARGET]

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = (
        r2_score(y, predictions)
        if len(y) > 1 and y.nunique() > 1
        else np.nan
    )

    benchmark_mae = mean_absolute_error(
        y, np.full(len(y), training_mean)
    )

    metrics = {
        "Dataset": dataset_name,
        "Flights": len(data),
        "MAE_L": mae,
        "RMSE_L": rmse,
        "R2": r2,
        "Training_mean_benchmark_MAE_L": benchmark_mae,
    }

    print(f"\n{dataset_name} performance (random forest)")
    print(f"MAE:  {mae:,.1f} L")
    print(f"RMSE: {rmse:,.1f} L")
    print(f"R²:   {r2:.3f}")
    print(f"Benchmark MAE: {benchmark_mae:,.1f} L")

    results = data.copy()
    results["Predicted Uplift (L)"] = predictions
    results["Error (L)"] = (
        results["Predicted Uplift (L)"] - results[TARGET]
    )
    results["Absolute Error (L)"] = results["Error (L)"].abs()

    prediction_file = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}_predictions_rf.xlsx"
    )

    results.to_excel(prediction_file, index=False)

    print(f"Predictions saved: {prediction_file}")

    return metrics


# ---------------------------------------------------------
# 5. Which inputs matter (permutation importance)
# ---------------------------------------------------------

def report_importance(model, data):
    """
    Shuffle one input at a time on validation data and measure how
    much MAE worsens. Works on the original columns, so each
    categorical feature gets one score rather than one per category.
    """
    result = permutation_importance(
        model,
        data[FEATURES],
        data[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "MAE_increase_L": result.importances_mean,
        "Std_L": result.importances_std,
    }).sort_values("MAE_increase_L", ascending=False)

    print("\nFeature importance (MAE increase when shuffled)")
    print(importance.round(1).to_string(index=False))

    importance.to_csv(
        OUTPUT_DIR / "feature_importance_rf.csv",
        index=False,
    )


# ---------------------------------------------------------
# 6. Compare with the linear model
# ---------------------------------------------------------

def compare_with_linear(rf_metrics):
    if not LINEAR_METRICS.exists():
        print(
            "\nNo linear results found. Run fuel_uplift_linear.py "
            "first to get a side-by-side comparison."
        )
        return

    linear = pd.read_csv(LINEAR_METRICS)
    rf = pd.DataFrame(rf_metrics)

    columns = ["Dataset", "MAE_L", "RMSE_L", "R2"]

    comparison = linear[columns].merge(
        rf[columns],
        on="Dataset",
        suffixes=("_Linear", "_RF"),
    )

    if comparison.empty:
        print(
            "\nLinear and RF runs share no datasets "
            "(e.g. one used --evaluate-test, the other didn't)."
        )
        return

    comparison["MAE_change_%"] = (
        comparison["MAE_L_RF"] / comparison["MAE_L_Linear"] - 1
    ) * 100

    print("\nLinear vs random forest")
    print(comparison.round(3).to_string(index=False))

    comparison.to_csv(
        OUTPUT_DIR / "linear_vs_rf.csv",
        index=False,
    )


# ---------------------------------------------------------
# 7. Main workflow
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a fuel uplift random forest."
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the finalized model on test.xlsx.",
    )

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

    model, best_params = tune_model(X_train, y_train, validation)

    training_mean = float(y_train.mean())

    metrics = [
        evaluate_model(model, validation, "Validation", training_mean)
    ]

    if args.evaluate_test:
        test = load_dataset(TEST_FILE)
        report_unfamiliar_categories(X_train, test[FEATURES])
        metrics.append(
            evaluate_model(model, test, "Test", training_mean)
        )

    pd.DataFrame(metrics).assign(
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=str(best_params["max_features"]),
    ).to_csv(
        OUTPUT_DIR / "evaluation_metrics_rf.csv",
        index=False,
    )

    compare_with_linear(metrics)

    report_importance(model, validation)

    model_path = OUTPUT_DIR / "rf_fuel_uplift_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")

    # Same illustrative flight as the linear script.
    new_flight = pd.DataFrame([{
        "Day of Week": "Monday",
        "Aircraft": "320",
        "Turboprop": "No",
        "To IATA": "DEL",
        "From Landed Price (Rs/KL)": 91974,
        "To Landed Price (Rs/KL)": 85358,
    }])

    new_flight = clean_features(new_flight)
    report_unfamiliar_categories(X_train, new_flight[FEATURES])

    loaded_model = joblib.load(model_path)
    predicted_uplift = loaded_model.predict(new_flight[FEATURES])[0]

    print(
        "\nIllustrative flight predicted uplift: "
        f"{predicted_uplift:,.0f} L"
    )


if __name__ == "__main__":
    main()