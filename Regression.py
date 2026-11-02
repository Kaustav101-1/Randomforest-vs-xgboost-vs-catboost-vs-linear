# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:49:44 2026

@author: Kaustav palit
"""

"""
Predict fuel uplift using linear regression.

Required Excel files:
    training.xlsx
    validation.xlsx
    test.xlsx

Required columns in each file:
    Day of Week
    Aircraft
    Turboprop
    To IATA
    From Landed Price (Rs/KL)
    To Landed Price (Rs/KL)
    Qty Uplifted (L)

Other columns are ignored.

Run:
    python fuel_uplift_linear.py

Final test evaluation:
    python fuel_uplift_linear.py --evaluate-test
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "training.xlsx"
VALIDATION_FILE = BASE_DIR / "validation.xlsx"
TEST_FILE = BASE_DIR / "test.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs"

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


# ---------------------------------------------------------
# 2. Clean input features
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
        raise ValueError(
            f"Missing input columns: {missing_columns}"
        )

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
        numbers[
            ~np.isfinite(numbers) | (numbers <= 0)
        ] = np.nan

        data[column] = numbers

    return data


# ---------------------------------------------------------
# 3. Load a labelled dataset
# ---------------------------------------------------------

def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    data = pd.read_excel(file_path)
    data.columns = data.columns.astype(str).str.strip()

    if data.columns.duplicated().any():
        raise ValueError(
            f"{file_path.name}: duplicate column headings."
        )

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

    invalid_target = (
        ~np.isfinite(data[TARGET])
        | (data[TARGET] < 0)
    )

    if invalid_target.any():
        # Excel row 1 contains the headings.
        excel_rows = (
            np.flatnonzero(invalid_target.to_numpy()) + 2
        )

        raise ValueError(
            f"{file_path.name}: invalid uplift in "
            f"{invalid_target.sum()} rows. "
            f"Example Excel rows: {excel_rows[:10].tolist()}"
        )

    if data.empty:
        raise ValueError(
            f"{file_path.name}: dataset is empty."
        )

    print(
        f"\nLoaded {file_path.name}: "
        f"{len(data):,} flights"
    )

    missing_counts = data[FEATURES].isna().sum()

    if missing_counts.any():
        print("Missing inputs to be imputed:")
        print(missing_counts[missing_counts > 0].to_string())

    return data


# ---------------------------------------------------------
# 4. Build the linear regression pipeline
# ---------------------------------------------------------

def build_model():
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
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ])

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
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

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression()),
    ])


# ---------------------------------------------------------
# 5. Report categories absent from training
# ---------------------------------------------------------

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
# 6. Evaluate and save predictions
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

    # Simple benchmark: always predict training-set mean.
    benchmark_predictions = np.full(
        len(y), training_mean
    )

    benchmark_mae = mean_absolute_error(
        y, benchmark_predictions
    )

    metrics = {
        "Dataset": dataset_name,
        "Flights": len(data),
        "MAE_L": mae,
        "RMSE_L": rmse,
        "R2": r2,
        "Training_mean_benchmark_MAE_L": benchmark_mae,
        "Negative_predictions": int(
            (predictions < 0).sum()
        ),
    }

    print(f"\n{dataset_name} performance")
    print(f"MAE:  {mae:,.1f} L")
    print(f"RMSE: {rmse:,.1f} L")
    print(f"R²:   {r2:.3f}")
    print(f"Benchmark MAE: {benchmark_mae:,.1f} L")
    print(
        "Negative predictions:",
        metrics["Negative_predictions"],
    )

    results = data.copy()

    results["Predicted Uplift (L)"] = predictions

    results["Error (L)"] = (
        results["Predicted Uplift (L)"]
        - results[TARGET]
    )

    results["Absolute Error (L)"] = (
        results["Error (L)"].abs()
    )

    prediction_file = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}_predictions.xlsx"
    )

    results.to_excel(prediction_file, index=False)

    print(f"Predictions saved: {prediction_file}")

    return metrics


# ---------------------------------------------------------
# 7. Main workflow
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a fuel uplift linear regression."
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the finalized model on test.xlsx.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load separate files. No automatic splitting.
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

    # Fit preprocessing and regression on training only.
    model = build_model()
    model.fit(X_train, y_train)

    print("\nLinear regression training completed.")

    training_mean = float(y_train.mean())

    report_unfamiliar_categories(
        X_train,
        validation[FEATURES],
    )

    metrics = [
        evaluate_model(
            model,
            validation,
            "Validation",
            training_mean,
        )
    ]

    # Test data is used only when explicitly requested.
    if args.evaluate_test:
        test = load_dataset(TEST_FILE)

        report_unfamiliar_categories(
            X_train,
            test[FEATURES],
        )

        metrics.append(
            evaluate_model(
                model,
                test,
                "Test",
                training_mean,
            )
        )

    pd.DataFrame(metrics).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv",
        index=False,
    )

    # Save the training-fitted model and its preprocessing.
    model_path = OUTPUT_DIR / "linear_fuel_uplift_model.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved: {model_path}")

    # Illustrative new flight: replace with actual inputs.
    new_flight = pd.DataFrame([{
        "Day of Week": "Monday",
        "Aircraft": "320",
        "Turboprop": "No",
        "To IATA": "DEL",
        "From Landed Price (Rs/KL)": 91974,
        "To Landed Price (Rs/KL)": 85358,
    }])

    new_flight = clean_features(new_flight)

    report_unfamiliar_categories(
        X_train,
        new_flight[FEATURES],
    )

    loaded_model = joblib.load(model_path)

    predicted_uplift = loaded_model.predict(
        new_flight[FEATURES]
    )[0]

    print(
        "\nIllustrative flight predicted uplift: "
        f"{predicted_uplift:,.0f} L"
    )

    if predicted_uplift < 0:
        print(
            "The prediction is negative; investigate model fit "
            "and these inputs before using the result."
        )


if __name__ == "__main__":
    main()