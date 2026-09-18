# 1-Year Indoor Environmental Time Series Forecasting

Deep learning models for predicting indoor temperature and humidity from two sensors, trained on one year of minute-level data.

## Dataset

- **File**: `dataclean_1year.csv` — 525,604 rows at 1-minute frequency
- **Features**: `date`, `hours`, `minutes`
- **Targets**: `temperatureC`, `humidityporc`, `temperatureC2`, `humidityporc2`
- **Source**: Two room sensors (sensor 1: temperatureC/humidityporc, sensor 2: temperatureC2/humidityporc2)

## Models

| Model | Directory | File |
|-------|-----------|------|
| CNN 1D | `1D/` | `best_conv1d_model_final.h5` |
| LSTM | `LSTM/` | `1yearbest_lstm_model.h5` |
| MLP | `MLP/` | `1yearbest_mlp_model_final.h5` |

Each model directory contains training checkpoints, CV logs, and cross-validation checkpoints.

## Scripts

- **`autofold.py`** — Automatic optimal fold detection for time series expanding-window cross-validation. Analyzes dataset frequency, detects gaps, and computes the maximum number of folds respecting minimum train size and validation window constraints.

- **`predictionsRange.py`** — Run inference over a user-defined date range using the CNN 1D model. Compares predictions against real sensor data (`habitacion.csv`), generates 4 separate plots (one per variable), and prints MAE, RMSE, correlation, and R² metrics.

## Preprocessing

`googleColab/preprocesamientodata.ipynb` — Hybrid imputation for missing sensor values, timestamp reconstruction, and data cleaning.

## Requirements

See `requirements.txt`. Built with TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib.


## 👨‍💻 Author / Autor

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)