import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# =============================================================================
# CONFIG
# =============================================================================

# Dataset ORIGINAL usado para entrenar
TRAIN_DATASET_PATH = "dataclean_1year.csv"

# Dataset REAL para comparar
COMPARE_DATASET_PATH = "habitacion.csv"

# Modelo .h5
MODEL_PATH = "C:\\Users\\user\\Desktop\\1year\\1D\\best_conv1d_model_final.h5"

SEQ_LEN = 60

OUTPUT_COLS = [
    "temperatureC",
    "humidityporc",
    "temperatureC2",
    "humidityporc2",
]

FEATURE_COLS = [
    "date",
    "hours",
    "minutes",
]

# =============================================================================
# HELPERS
# =============================================================================

def to_python_ordinal(date_str):

    return datetime.strptime(
        date_str,
        "%Y-%m-%d"
    ).toordinal()


def build_datetime(date_str, hour, minute):

    return datetime.strptime(
        f"{date_str} {hour:02d}:{minute:02d}",
        "%Y-%m-%d %H:%M"
    )


def datetime_to_parts(dt):

    return (
        dt.strftime("%Y-%m-%d"),
        dt.hour,
        dt.minute
    )


def build_datetime_column(df):

    df["datetime_real"] = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta(df["hours"], unit="h")
        + pd.to_timedelta(df["minutes"], unit="m")
    )

    return df


def run_inference(
    model,
    scaler_output,
    window60,
    feat_scaled
):

    seq_input = np.expand_dims(
        window60,
        axis=0
    )  # (1,60,4)

    feat_input = np.array(
        feat_scaled
    )  # (1,3)

    pred_scaled = model.predict(
        [seq_input, feat_input],
        verbose=0
    )

    pred_original = scaler_output.inverse_transform(
        pred_scaled
    )

    return pred_scaled[0], pred_original[0]


# =============================================================================
# LOAD TRAIN DATASET
# =============================================================================

print("=" * 60)
print("LOADING TRAIN DATASET")
print("=" * 60)

train_df = pd.read_csv(TRAIN_DATASET_PATH)

if "id" in train_df.columns:
    train_df = train_df.drop("id", axis=1)

train_df = build_datetime_column(train_df)

train_df = train_df.sort_values(
    "datetime_real"
).reset_index(drop=True)

# ordinal date
train_df["date"] = pd.to_datetime(
    train_df["date"]
).apply(lambda x: x.toordinal())

# =============================================================================
# SCALERS
# =============================================================================

print("\nFitting scalers...")

scaler_output = MinMaxScaler()
scaler_features = MinMaxScaler()

output_scaled = scaler_output.fit_transform(
    train_df[OUTPUT_COLS]
)

features_scaled = scaler_features.fit_transform(
    train_df[FEATURE_COLS]
)

# =============================================================================
# ARRAYS
# =============================================================================

datetime_arr = train_df["datetime_real"].values

output_raw_full = train_df[OUTPUT_COLS].values

output_scaled_full = output_scaled

dataset_start = train_df["datetime_real"].iloc[0]
dataset_end   = train_df["datetime_real"].iloc[-1]

N = len(train_df)

print(f"\nDataset rows: {N}")

print("\nDataset range:")
print(dataset_start, "->", dataset_end)

# =============================================================================
# LOAD MODEL
# =============================================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")

# =============================================================================
# LOAD COMPARISON DATASET
# =============================================================================

print("\n" + "=" * 60)
print("LOADING COMPARISON DATASET")
print("=" * 60)

compare_df = pd.read_csv(COMPARE_DATASET_PATH)

if "id" in compare_df.columns:
    compare_df = compare_df.drop("id", axis=1)

compare_df = build_datetime_column(compare_df)

compare_df = compare_df.sort_values(
    "datetime_real"
).reset_index(drop=True)

print("✅ Comparison dataset loaded")

print("\nComparison dataset range:")
print(
    compare_df["datetime_real"].iloc[0],
    "->",
    compare_df["datetime_real"].iloc[-1]
)

# =============================================================================
# USER INPUT
# =============================================================================

print("\n" + "=" * 60)
print("RANGE PREDICTION")
print("=" * 60)

start_date_str = input(
    "Enter the start date for prediction (YYYY-MM-DD): "
)

start_hour = int(
    input("Enter the start hour for prediction (0-23): ")
)

start_minute = int(
    input("Enter the start minute for prediction (0-59): ")
)

end_date_str = input(
    "Enter the end date for prediction (YYYY-MM-DD): "
)

end_hour = int(
    input("Enter the end hour for prediction (0-23): ")
)

end_minute = int(
    input("Enter the end minute for prediction (0-59): ")
)

# =============================================================================
# BUILD RANGE
# =============================================================================

start_dt = build_datetime(
    start_date_str,
    start_hour,
    start_minute
)

end_dt = build_datetime(
    end_date_str,
    end_hour,
    end_minute
)

if start_dt > end_dt:
    raise ValueError(
        "Start datetime must be <= end datetime"
    )

# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

print("\nRunning predictions...")

prediction_points = []

cur_dt = start_dt

while cur_dt <= end_dt:

    # =========================================================================
    # FUTURE PREDICTION USING ANCHORED SLIDING WINDOW
    # =========================================================================

    future_start = max(
        start_dt,
        dataset_end + timedelta(minutes=1)
    )

    future_steps_total = int(
        (end_dt - future_start).total_seconds() / 60
    ) + 1

    j = int(
        (cur_dt - future_start).total_seconds() / 60
    )

    anchor_idx = max(
        SEQ_LEN,
        N - future_steps_total + j
    )

    window_end = min(
        anchor_idx,
        N
    )

    window_start = max(
        0,
        window_end - SEQ_LEN
    )

    window60 = output_scaled_full[
        window_start:window_end
    ]

    while len(window60) < SEQ_LEN:

        window60 = np.vstack([
            output_scaled_full[0],
            window60
        ])

    date_str, h, m = datetime_to_parts(
        cur_dt
    )

    ordinal = to_python_ordinal(
        date_str
    )

    feat_scaled = scaler_features.transform(

        pd.DataFrame(
            [[ordinal, h, m]],
            columns=FEATURE_COLS
        )

    )

    _, pred_original = run_inference(
        model,
        scaler_output,
        window60,
        feat_scaled
    )

    prediction_points.append({

        "datetime": cur_dt,

        "temperatureC": pred_original[0],
        "humidityporc": pred_original[1],
        "temperatureC2": pred_original[2],
        "humidityporc2": pred_original[3],

    })

    cur_dt += timedelta(minutes=1)

# =============================================================================
# PREDICTIONS DATAFRAME
# =============================================================================

pred_df = pd.DataFrame(
    prediction_points
)

print("\n✅ Predictions completed")

# =============================================================================
# FILTER COMPARISON DATASET
# =============================================================================

compare_range_df = compare_df[

    (compare_df["datetime_real"] >= start_dt)
    &
    (compare_df["datetime_real"] <= end_dt)

].copy()

print("\nComparison points:", len(compare_range_df))
print("Prediction points:", len(pred_df))

# =============================================================================
# MERGE
# =============================================================================

merged_df = pd.merge(

    pred_df,

    compare_range_df[
        ["datetime_real"] + OUTPUT_COLS
    ],

    left_on="datetime",
    right_on="datetime_real",
    how="left",

    suffixes=("_pred", "_real")

)

# =============================================================================
# SAVE CSV
# =============================================================================

SAVE_PATH = "prediction_vs_real.csv"

merged_df.to_csv(
    SAVE_PATH,
    index=False
)

print("\n✅ CSV saved:")
print(SAVE_PATH)

# =============================================================================
# 4 SEPARATED PLOTS
# =============================================================================

print("\nGenerating plots...")

variables = [

    ("temperatureC",  "Temperature C"),
    ("humidityporc",  "Humidity %"),
    ("temperatureC2", "Temperature C2"),
    ("humidityporc2", "Humidity % 2"),

]

# =============================================================================
# CREATE ONE FIGURE PER VARIABLE
# =============================================================================

for var, label in variables:

    plt.figure(figsize=(20, 7))

    # =========================================================================
    # REAL VALUES
    # =========================================================================

    plt.plot(

        compare_range_df["datetime_real"],
        compare_range_df[var],

        linewidth=2,

        label=f"{label} REAL"

    )

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    plt.plot(

        pred_df["datetime"],
        pred_df[var],

        linestyle="--",
        linewidth=2,

        label=f"{label} PRED"

    )

    # =========================================================================
    # TITLE
    # =========================================================================

    plt.title(
        f"CNN1D Prediction vs Real - {label}",
        fontsize=18
    )

    # =========================================================================
    # LABELS
    # =========================================================================

    plt.xlabel(
        "Datetime",
        fontsize=13
    )

    plt.ylabel(
        label,
        fontsize=13
    )

    # =========================================================================
    # GRID
    # =========================================================================

    plt.grid(True)

    # =========================================================================
    # LEGEND
    # =========================================================================

    plt.legend()

    # =========================================================================
    # LAYOUT
    # =========================================================================

    plt.tight_layout()

    # =========================================================================
    # SHOW
    # =========================================================================

    plt.show()

# =============================================================================
# METRICS
# =============================================================================

print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)

for var in OUTPUT_COLS:

    valid = merged_df[
        [f"{var}_pred", f"{var}_real"]
    ].dropna()

    if len(valid) == 0:
        continue

    y_true = valid[f"{var}_real"].values
    y_pred = valid[f"{var}_pred"].values

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )

    corr = np.corrcoef(
        y_true,
        y_pred
    )[0, 1]

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n[{var}]")
    print(f"MAE :  {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"CORR:  {corr:.4f}")
    print(f"R²  :  {r2:.4f}")

print("\n✅ Done")