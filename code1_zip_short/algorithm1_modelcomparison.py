# =============================================================================
# Algorithm1_ModelComparison.py
#
# Wildfire PM2.5 Calibration and Model Comparison
#
# Purpose:
#   Compare a log-linear calibration model against a multilayer perceptron (MLP)
#   for PurpleAir PM2.5 calibration against an EPA reference monitor.
#
# Method:
#   - Full-data log-linear calibration fit
#   - 50 repeated random train/test splits
#   - OLS vs MLP R2 comparison
#   - Publication-quality diagnostic figures
#
# Author:
#   Generated for thesis/research workflow
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
import glob
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy import stats

from tqdm import tqdm


# TensorFlow / Keras
import tensorflow as tf

from tensorflow.keras import (
    Sequential,
    layers,
    callbacks,
    optimizers
)


# Ignore harmless warnings
warnings.filterwarnings(
    "ignore"
)



# =============================================================================
# CONFIGURATION
# =============================================================================


# Main directory containing all input files
BASE_DIR = os.getcwd()


# Input files
EPA_FILE = os.path.join(
    BASE_DIR,
    "LA_Site_1103.csv"
)

COORD_FILE = os.path.join(
    BASE_DIR,
    "Sensors_with_coordinates.csv"
)


# PurpleAir sensor files
PURPLEAIR_PATTERN = os.path.join(
    BASE_DIR,
    "sensor_*_history.csv"
)


# Required calibration sensor
CALIBRATION_SENSOR = "208493"



# Output directories

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

CALIBRATION_DIR = os.path.join(
    OUTPUT_DIR,
    "calibration"
)


FIGURE_DIR = os.path.join(
    CALIBRATION_DIR,
    "figures"
)



# Create folders if they do not exist

os.makedirs(
    CALIBRATION_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)



# =============================================================================
# LOGGING SETUP
# =============================================================================


LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "algorithm1_log.txt"
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE
        ),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger()



# =============================================================================
# RANDOM SEED CONTROL
# =============================================================================

# Algorithm 1 requires reproducibility

RANDOM_SEED = 42


np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


logger.info(
    "Random seeds initialized to 42"
)



# =============================================================================
# PURPLEAIR DATA LOADING
# =============================================================================


def load_purpleair_sensor_file(
    filepath
):
    """
    Load a single PurpleAir history CSV.

    Parameters
    ----------
    filepath : str
        Path to PurpleAir csv file.

    Returns
    -------
    pandas.DataFrame
        Cleaned hourly PurpleAir observations.
    """


    sensor_id = (
        os.path.basename(filepath)
        .replace(
            "sensor_",
            ""
        )
        .replace(
            "_history.csv",
            ""
        )
    )


    try:

        df = pd.read_csv(
            filepath
        )


        required_columns = [
            "time_stamp",
            "pm2.5_cf_1",
            "humidity",
            "temperature",
            "pressure"
        ]


        missing = [
            c for c in required_columns
            if c not in df.columns
        ]


        if missing:

            logger.warning(
                f"{sensor_id} missing columns: {missing}"
            )

            return None



        df = df[
            required_columns
        ].copy()



        df["timestamp"] = pd.to_datetime(
            df["time_stamp"],
            errors="coerce"
        )


        df = df.dropna(
            subset=[
                "timestamp"
            ]
        )


        df["sensor"] = sensor_id



        df = df.rename(
            columns={
                "pm2.5_cf_1": "pm25"
            }
        )


        return df[
            [
                "sensor",
                "timestamp",
                "pm25",
                "humidity",
                "temperature",
                "pressure"
            ]
        ]



    except Exception as e:

        logger.exception(
            f"Failed loading {filepath}: {e}"
        )

        return None
# =============================================================================
# LOAD ALL PURPLEAIR SENSORS
# =============================================================================


def load_all_purpleair_sensors():
    """
    Load all PurpleAir sensor history files.

    Sensors with fewer than 120 hourly observations
    are discarded.

    Returns
    -------
    pandas.DataFrame
        Combined PurpleAir observations.
    """


    logger.info(
        "Searching for PurpleAir files..."
    )


    files = glob.glob(
        PURPLEAIR_PATTERN
    )


    logger.info(
        f"Found {len(files)} PurpleAir files"
    )


    sensor_data = []


    for file in tqdm(
        files,
        desc="Loading PurpleAir sensors"
    ):

        df = load_purpleair_sensor_file(
            file
        )


        if df is None:
            continue


        sensor_id = df["sensor"].iloc[0]


        if len(df) < 120:

            logger.info(
                f"Discarding sensor {sensor_id} "
                f"({len(df)} observations)"
            )

            continue


        sensor_data.append(
            df
        )



    if len(sensor_data) == 0:

        raise ValueError(
            "No valid PurpleAir sensors found."
        )


    combined = pd.concat(
        sensor_data,
        ignore_index=True
    )


    logger.info(
        f"Retained {combined['sensor'].nunique()} PurpleAir sensors"
    )


    return combined



# =============================================================================
# LOAD EPA REFERENCE MONITOR
# =============================================================================


def load_epa_monitor():
    """
    Load EPA reference monitor data.

    Uses:
        Date Local
        Time Local

    Keeps:
        2025-01-01 00:00
        through
        2025-02-28 23:00

    No interpolation is performed.

    Returns
    -------
    pandas.DataFrame
        Hourly EPA PM2.5 observations.
    """


    logger.info(
        "Loading EPA reference monitor data..."
    )


    epa = pd.read_csv(
        EPA_FILE
    )


    required = [
        "Date Local",
        "Time Local",
        "Sample Measurement"
    ]


    missing = [
        c for c in required
        if c not in epa.columns
    ]


    if missing:

        raise ValueError(
            f"EPA file missing columns: {missing}"
        )



    # Construct local timestamp
    epa["timestamp"] = pd.to_datetime(
        epa["Date Local"].astype(str)
        + " "
        + epa["Time Local"].astype(str),
        errors="coerce"
    )


    epa = epa.dropna(
        subset=[
            "timestamp"
        ]
    )


    start = pd.Timestamp(
        "2025-01-01 00:00:00"
    )

    end = pd.Timestamp(
        "2025-02-28 23:00:00"
    )


    epa = epa[
        (epa["timestamp"] >= start)
        &
        (epa["timestamp"] <= end)
    ]



    epa = epa[
        [
            "timestamp",
            "Sample Measurement"
        ]
    ].copy()



    epa = epa.rename(
        columns={
            "Sample Measurement": "epa_pm25"
        }
    )


    logger.info(
        f"EPA observations retained: {len(epa)}"
    )


    return epa



# =============================================================================
# SYNCHRONIZE EPA WITH CALIBRATION SENSOR
# =============================================================================


def synchronize_calibration_data(
    purpleair,
    epa
):
    """
    Synchronize EPA and PurpleAir sensor 208493.

    Only timestamps existing in both datasets
    are retained.

    No interpolation is performed.

    Returns
    -------
    pandas.DataFrame
        Matched calibration dataset.
    """


    logger.info(
        "Synchronizing EPA with PurpleAir "
        f"sensor {CALIBRATION_SENSOR}"
    )


    pa = purpleair[
        purpleair["sensor"]
        ==
        CALIBRATION_SENSOR
    ].copy()



    if len(pa) == 0:

        raise ValueError(
            f"Calibration sensor {CALIBRATION_SENSOR} not found."
        )



    merged = pd.merge(
        epa,
        pa,
        on="timestamp",
        how="inner"
    )



    logger.info(
        f"Synchronized observations: {len(merged)}"
    )



    merged = merged.sort_values(
        "timestamp"
    )



    merged.to_csv(
        os.path.join(
            CALIBRATION_DIR,
            "synchronized_calibration_data.csv"
        ),
        index=False
    )


    return merged



# =============================================================================
# CREATE LAG VARIABLES
# =============================================================================


def create_lag_features(
    dataframe
):
    """
    Create one-hour lagged meteorological predictors.

    Adds:

        temperature_lag1
        humidity_lag1
        pressure_lag1

    First observation is removed.

    Returns
    -------
    pandas.DataFrame
    """


    df = dataframe.copy()


    df = df.sort_values(
        "timestamp"
    )


    df[
        "temperature_lag1"
    ] = df[
        "temperature"
    ].shift(1)


    df[
        "humidity_lag1"
    ] = df[
        "humidity"
    ].shift(1)


    df[
        "pressure_lag1"
    ] = df[
        "pressure"
    ].shift(1)



    df = df.dropna(
        subset=[
            "temperature_lag1",
            "humidity_lag1",
            "pressure_lag1"
        ]
    )


    logger.info(
        f"Observations after lag construction: {len(df)}"
    )


    return df
# =============================================================================
# MODEL PREDICTOR DEFINITIONS
# =============================================================================


PREDICTORS = [
    "pm25",
    "temperature",
    "humidity",
    "pressure",
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1"
]



TARGET = "epa_pm25"



# =============================================================================
# PREPARE MODEL MATRICES
# =============================================================================


def prepare_model_data(
    dataframe
):
    """
    Prepare predictors and target variables.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Calibration dataset.

    Returns
    -------
    X : pandas.DataFrame
        Predictor matrix.

    y : pandas.Series
        EPA PM2.5 target.
    """


    df = dataframe.copy()


    # Remove invalid PM values
    df = df[
        (df["pm25"] > 0)
        &
        (df["epa_pm25"] > 0)
    ]


    X = df[
        PREDICTORS
    ].copy()


    y = df[
        TARGET
    ].copy()



    logger.info(
        f"Final calibration rows: {len(df)}"
    )


    return X, y, df



# =============================================================================
# LOG-LINEAR CALIBRATION MODEL
# =============================================================================


def fit_loglinear_model(
    dataframe
):
    """
    Fit full-data log-linear calibration model.

    Equation:

    log(EPA PM2.5) =
        beta0
        + beta1 log(PurpleAir PM2.5)
        + beta2 Temperature
        + beta3 Humidity
        + beta4 Pressure
        + beta5 Lag Temperature
        + beta6 Lag Humidity
        + beta7 Lag Pressure


    Returns
    -------
    model
        Statsmodels OLS model.
    """


    df = dataframe.copy()


    X = pd.DataFrame()


    X["log_pm25"] = np.log(
        df["pm25"]
    )


    X[
        "temperature"
    ] = df[
        "temperature"
    ]


    X[
        "humidity"
    ] = df[
        "humidity"
    ]


    X[
        "pressure"
    ] = df[
        "pressure"
    ]


    X[
        "temperature_lag1"
    ] = df[
        "temperature_lag1"
    ]


    X[
        "humidity_lag1"
    ] = df[
        "humidity_lag1"
    ]


    X[
        "pressure_lag1"
    ] = df[
        "pressure_lag1"
    ]



    X = sm.add_constant(
        X
    )


    y = np.log(
        df["epa_pm25"]
    )



    model = sm.OLS(
        y,
        X
    ).fit()



    logger.info(
        "Log-linear model fitted."
    )


    logger.info(
        f"Training R2 = {model.rsquared:.4f}"
    )



    return model



# =============================================================================
# SAVE LOG-LINEAR COEFFICIENTS
# =============================================================================


def save_loglinear_coefficients(
    model
):
    """
    Save log-linear regression coefficients.
    """


    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values
        }
    )


    output = os.path.join(
        CALIBRATION_DIR,
        "loglinear_coefficients.csv"
    )


    coefficients.to_csv(
        output,
        index=False
    )


    logger.info(
        f"Saved log-linear coefficients: {output}"
    )



# =============================================================================
# BUILD MLP MODEL
# =============================================================================


def build_mlp_model(
    input_shape
):
    """
    Construct MLP calibration model.

    Architecture:

        Dense(128, ReLU)
        BatchNormalization
        Dropout(0.3)
        Dense(64, ReLU)
        Dense(32, ReLU)
        Dense(1)


    Parameters
    ----------
    input_shape : int
        Number of predictors.

    Returns
    -------
    tensorflow.keras.Model
    """


    model = Sequential(
        [

            layers.Input(
                shape=(input_shape,)
            ),


            layers.Dense(
                128,
                activation="relu"
            ),


            layers.BatchNormalization(),


            layers.Dropout(
                0.3
            ),


            layers.Dense(
                64,
                activation="relu"
            ),


            layers.Dense(
                32,
                activation="relu"
            ),


            layers.Dense(
                1
            )

        ]
    )



    model.compile(
        optimizer=optimizers.Adam(),
        loss="mean_squared_error"
    )


    return model



# =============================================================================
# TRAIN MLP MODEL
# =============================================================================


def train_mlp_model(
    X_train,
    y_train,
    X_validation,
    y_validation
):
    """
    Train MLP with early stopping.

    Returns
    -------
    model
    history
    scaler
    """


    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )


    X_validation_scaled = scaler.transform(
        X_validation
    )



    model = build_mlp_model(
        X_train.shape[1]
    )



    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )



    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(
            X_validation_scaled,
            y_validation
        ),
        epochs=300,
        batch_size=32,
        callbacks=[
            early_stop
        ],
        verbose=0
    )



    return (
        model,
        history,
        scaler
    )
# =============================================================================
# RANDOM TRAIN/TEST MODEL COMPARISON
# =============================================================================


def run_random_model_comparison(
    dataframe,
    iterations=50
):
    """
    Compare log-linear and MLP models using repeated
    random train/test splits.

    Both models receive:
        - identical predictors
        - identical train/test split
        - identical test observations

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Calibration dataset.

    iterations : int
        Number of random train/test repetitions.

    Returns
    -------
    pandas.DataFrame
        R2 results for both models.
    """


    X, y, _ = prepare_model_data(
        dataframe
    )


    results = []



    logger.info(
        "Starting random train/test comparison..."
    )



    for i in tqdm(
        range(iterations),
        desc="Random CV iterations"
    ):


        logger.info(
            f"Iteration {i+1}/{iterations}"
        )



        # -------------------------------------------------------------
        # Same random split for both models
        # -------------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_SEED + i,
            shuffle=True
        )



        # -------------------------------------------------------------
        # LOG-LINEAR MODEL
        # -------------------------------------------------------------


        train_df = X_train.copy()

        train_df["epa_pm25"] = y_train.values


        ols_model = fit_loglinear_model(
            train_df
        )



        X_test_ols = pd.DataFrame()


        X_test_ols["log_pm25"] = np.log(
            X_test["pm25"]
        )


        for column in [
            "temperature",
            "humidity",
            "pressure",
            "temperature_lag1",
            "humidity_lag1",
            "pressure_lag1"
        ]:

            X_test_ols[column] = X_test[column]



        X_test_ols = sm.add_constant(
            X_test_ols,
            has_constant="add"
        )



        ols_prediction = np.exp(
            ols_model.predict(
                X_test_ols
            )
        )



        ols_r2 = r2_score(
            y_test,
            ols_prediction
        )



        # -------------------------------------------------------------
        # MLP MODEL
        # -------------------------------------------------------------


        X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=RANDOM_SEED + i
        )



        mlp_model, history, scaler = train_mlp_model(
            X_train_mlp,
            y_train_mlp,
            X_val_mlp,
            y_val_mlp
        )



        X_test_scaled = scaler.transform(
            X_test
        )


        mlp_prediction = mlp_model.predict(
            X_test_scaled,
            verbose=0
        ).flatten()



        mlp_r2 = r2_score(
            y_test,
            mlp_prediction
        )



        results.append(
            {
                "iteration": i + 1,
                "OLS_R2": ols_r2,
                "MLP_R2": mlp_r2
            }
        )



    results = pd.DataFrame(
        results
    )



    output = os.path.join(
        CALIBRATION_DIR,
        "random_cv_results.csv"
    )


    results.to_csv(
        output,
        index=False
    )


    logger.info(
        f"Saved random CV results: {output}"
    )



    return results




# =============================================================================
# FULL-DATA CALIBRATED PREDICTIONS
# =============================================================================


def generate_calibrated_predictions(
    dataframe,
    model
):
    """
    Generate calibrated PurpleAir PM2.5 values
    using the full-data log-linear model.

    These predictions are carried forward
    to Algorithm 2.

    Returns
    -------
    pandas.DataFrame
    """


    df = dataframe.copy()



    X = pd.DataFrame()


    X["log_pm25"] = np.log(
        df["pm25"]
    )


    for column in [
        "temperature",
        "humidity",
        "pressure",
        "temperature_lag1",
        "humidity_lag1",
        "pressure_lag1"
    ]:

        X[column] = df[column]



    X = sm.add_constant(
        X,
        has_constant="add"
    )



    df["calibrated_pm25"] = np.exp(
        model.predict(
            X
        )
    )



    output = os.path.join(
        CALIBRATION_DIR,
        "calibrated_training_predictions.csv"
    )



    df.to_csv(
        output,
        index=False
    )


    logger.info(
        f"Saved calibrated predictions: {output}"
    )



    return df
# =============================================================================
# FIGURE GENERATION
# =============================================================================


def plot_model_comparison(
    results
):
    """
    Create OLS vs MLP R2 boxplot.
    """

    plt.figure(
        figsize=(8, 6)
    )


    sns.boxplot(
        data=results[
            [
                "OLS_R2",
                "MLP_R2"
            ]
        ]
    )


    plt.ylabel(
        "Test R²"
    )


    plt.title(
        "Random Train/Test Model Comparison"
    )


    plt.tight_layout()


    output = os.path.join(
        FIGURE_DIR,
        "OLS_vs_MLP_R2_boxplot.png"
    )


    plt.savefig(
        output,
        dpi=300
    )


    plt.close()



def plot_calibration_scatter(
    dataframe,
    model
):
    """
    Create EPA vs PurpleAir and EPA vs calibrated
    PurpleAir scatter plots.
    """


    df = dataframe.copy()



    # Uncalibrated PurpleAir

    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        df["pm25"],
        df["epa_pm25"],
        alpha=0.5
    )


    plt.xlabel(
        "PurpleAir PM2.5"
    )


    plt.ylabel(
        "EPA PM2.5"
    )


    plt.title(
        "EPA vs PurpleAir PM2.5"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "EPA_vs_PurpleAir.png"
        ),
        dpi=300
    )


    plt.close()



    # Calibrated prediction

    calibrated = generate_calibrated_predictions(
        dataframe,
        model
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        calibrated["calibrated_pm25"],
        calibrated["epa_pm25"],
        alpha=0.5
    )


    plt.xlabel(
        "Calibrated PurpleAir PM2.5"
    )


    plt.ylabel(
        "EPA PM2.5"
    )


    plt.title(
        "EPA vs Calibrated PurpleAir PM2.5"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "EPA_vs_Calibrated_PurpleAir.png"
        ),
        dpi=300
    )


    plt.close()



    return calibrated



def plot_residuals(
    dataframe,
    model
):
    """
    Generate residual histogram and QQ plot.
    """


    calibrated = dataframe.copy()



    residuals = (
        calibrated["epa_pm25"]
        -
        calibrated["calibrated_pm25"]
    )



    # Histogram

    plt.figure(
        figsize=(7, 5)
    )


    plt.hist(
        residuals,
        bins=30
    )


    plt.xlabel(
        "Residual (EPA - Predicted)"
    )


    plt.ylabel(
        "Frequency"
    )


    plt.title(
        "Calibration Residual Distribution"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "residual_histogram.png"
        ),
        dpi=300
    )


    plt.close()



    # QQ plot

    plt.figure(
        figsize=(7, 5)
    )


    stats.probplot(
        residuals,
        dist="norm",
        plot=plt
    )


    plt.title(
        "Residual QQ Plot"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "residual_QQ_plot.png"
        ),
        dpi=300
    )


    plt.close()



# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():

    start_time = datetime.now()


    logger.info(
        "================================================="
    )

    logger.info(
        "Starting Algorithm 1: Model Comparison Pipeline"
    )

    logger.info(
        f"Start time: {start_time}"
    )

    logger.info(
        "================================================="
    )



    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    purpleair = load_all_purpleair_sensors()


    epa = load_epa_monitor()



    calibration = synchronize_calibration_data(
        purpleair,
        epa
    )



    calibration = create_lag_features(
        calibration
    )



    calibration.to_csv(
        os.path.join(
            CALIBRATION_DIR,
            "calibration_dataset.csv"
        ),
        index=False
    )



    # -------------------------------------------------------------------------
    # Full-data log-linear calibration
    # -------------------------------------------------------------------------

    loglinear_model = fit_loglinear_model(
        calibration
    )


    save_loglinear_coefficients(
        loglinear_model
    )



    calibrated = generate_calibrated_predictions(
        calibration,
        loglinear_model
    )



    # -------------------------------------------------------------------------
    # Model comparison
    # -------------------------------------------------------------------------

    comparison_results = run_random_model_comparison(
        calibration,
        iterations=50
    )



    comparison_results.to_csv(
        os.path.join(
            CALIBRATION_DIR,
            "model_comparison.csv"
        ),
        index=False
    )



    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    plot_model_comparison(
        comparison_results
    )


    plot_calibration_scatter(
        calibration,
        loglinear_model
    )


    plot_residuals(
        calibrated,
        loglinear_model
    )



    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    end_time = datetime.now()


    logger.info(
        "================================================="
    )

    logger.info(
        "Algorithm 1 completed successfully."
    )

    logger.info(
        f"End time: {end_time}"
    )


    logger.info(
        f"Runtime: {end_time-start_time}"
    )


    logger.info(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


    logger.info(
        "================================================="
    )



# =============================================================================
# RUN SCRIPT
# =============================================================================


if __name__ == "__main__":

    main()