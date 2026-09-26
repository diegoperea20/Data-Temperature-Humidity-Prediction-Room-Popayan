#  Data Temperature Humidity Prediction Room Popayan 

Multivariate time series forecasting system for **indoor temperature and humidity** measured by two sensors in a room in Popayan, Colombia. The project compares deep learning architectures (**MLP, LSTM and 1D CNN**) on two data horizons: **4 months** and **1 year**, with per-minute sampling and 60-minute ahead prediction.

> 📄 Spanish version: [README.es.md](README.es.md) | 📁 Modules: [`1year/`](1year/) · [`4months/`](4months/)


##### Evaluation of architecture configurations evaluated on 4-month dataset 


| Architecture | Score | Average R² | STD(R²) | Average MAE | Average Gap | Status |
|---|---:|---:|---:|---:|---:|---|
| Conv1D | 0.9000 | 0.9435 | 0.0168 | 0.8608 | 0.2483 | Selected |
| LSTM | -999 | 0.5734 | 0.3258 | 2.7277 | 1.3443 | Discarded |
| MLP | -999 | 0.5053 | 0.3932 | 2.9473 | 1.3689 | Discarded |





##### Evaluation of architecture configurations evaluated on 1-year dataset


| Architecture | Score | Average R² | STD(R²) | Average MAE | Average Gap | Status |
|---|---:|---:|---:|---:|---:|---|
| Conv1D (Conv1D_Original) | 0.9000 | 0.9481 | 0.0235 | 0.7997 | 0.2231 | Selected |
| LSTM (LSTM_Original) | -999 | 0.3079 | 0.4864 | 3.2043 | 2.1986 | Discarded |
| MLP (MLP_Light_Dropout) | -999 | 0.2164 | 0.7123 | 3.5066 | 0.5221 | Discarded |


#### Comparison of selected architectures from 4-month dataset with 1-year dataset 

| Metric | CONV1D 4 MONTHS | CONV1D 1 YEAR | 
|---|---:|---:|
| Average R² | 0.9435 | 0.9481 | 
| MAE| 0.8608 | 0.7997 | 
| GAP | 0.2483 | 0.2231 | 
| FINAL GAP| 1.1581 | 0.0427 |


----

### Dataset


| id      | date       | hours | minutes | temperatureC | humidityporc | temperatureC2 | humidityporc2 |
|---------|------------|-------|---------|--------------|--------------|---------------|---------------|
| 4776    | 2025-05-01 | 20    | 0       | 23.0         | 70           | 20.8          | 75            |
| 4777    | 2025-05-01 | 20    | 1       | 22.0         | 71           | 20.9          | 75            |
| 4778    | 2025-05-01 | 20    | 2       | 22.0         | 71           | 20.9          | 74            |
| 4779    | 2025-05-01 | 20    | 3       | 22.0         | 71           | 20.9          | 74            |
| ...     | ...        | ...   | ...     | ...          | ...          | ...           | ...           |
| 533046  | 2026-05-01 | 19    | 59      | 24.8         | 62           | 22.2          | 63            |
| 533047  | 2026-05-01 | 20    | 0       | 24.8         | 62           | 22.2          | 63            |
| 533048  | 2026-05-01 | 20    | 1       | 24.8         | 62           | 22.2          | 63            |
| 533049  | 2026-05-01 | 20    | 2       | 24.8         | 62           | 22.2          | 63            |



## Dataset acquisition and web application repositories 

[ESP32-DHT11-to-Supabase](https://github.com/diegoperea20/ESP32-DHT11-to-Supabase)

[Web Temperature-Humidity-Prediction-Room-Popayan](https://github.com/diegoperea20/Temperature-Humidity-Prediction-Room-Popayan)



---

## 📑 Table of contents

- [Popayan Room Indoor Temperature and Humidity Prediction Data](#popayan-room-indoor-temperature-and-humidity-prediction-data)
        - [](#)
    - [Dataset](#dataset)
  - [📑 Table of contents](#-table-of-contents)
  - [1. Project description](#1-project-description)
  - [3. Objectives](#3-objectives)
  - [4. General repository structure](#4-general-repository-structure)
  - [5. Datasets](#5-datasets)
  - [6. Methodology](#6-methodology)
  - [7. Compared models](#7-compared-models)
  - [8. Main results](#8-main-results)
  - [9. Scripts and utilities](#9-scripts-and-utilities)
  - [10. Installation](#10-installation)
  - [11. Usage](#11-usage)
  - [12. Technical requirements](#12-technical-requirements)
  - [13. Conclusions](#13-conclusions)
  - [14. Future work](#14-future-work)
    - [📄 License](#-license)
  - [👨‍💻 Author](#-author)

---

## 1. Project description

This repository contains the complete development of an indoor environmental prediction system, within the framework of master's training in data science. From minute-by-minute records of **two sensors** (temperature in °C and relative humidity in %), neural models capable of anticipating the thermal and humidity behavior of a room are trained and evaluated.

The work is organized into two complementary studies:

| Module | Dataset | Temporal coverage | Purpose |
|--------|---------|-------------------|-----------|
| [`4months/`](4months/) | `dataclean_4months.csv` | ~4 months since May 2025, 177,121 records | Controlled study and rigorous comparison MLP vs. LSTM vs. 1D CNN |
| [`1year/`](1year/) | `dataclean_1year.csv` | 1 full year, 525,604 records | Long-term study: seasonality, robustness and inference deployment by date ranges |

Both modules share the same formulation: **60-minute history sliding window → 60-minute ahead forecast** for the 4 target variables.

**Target variables (targets):**

- `temperatureC` / `humidityporc` — sensor 1
- `temperatureC2` / `humidityporc2` — sensor 2

See detail in [`1year/README.es.md`](1year/README.es.md) and [`4months/README.es.md`](4months/README.es.md).


---

## 3. Objectives

**General objective:**

To develop a predictive model based on machine learning to estimate the air temperature and humidity in a residential dwelling room in humid climatic contexts, in order to anticipate comfort conditions and prevent health risks.

**Specific objectives:**

• To characterize the humidity and temperature behavior of the study room through the construction and analysis of a time series dataset.

• To select a set of candidate models, based on a state-of-the-art review that are relevant to the problem of joint estimation of environmental variables.

• To implement an interactive visualization interface prototype that presents the predictions generated by the implemented models.

• To comparatively evaluate the performance of the selected models, using a validation protocol and quantitative metrics suitable for regression problems, in order to determine the architecture with the best balance between accuracy and efficiency.


---

## 4. General repository structure

```text
maestriadata1year4mo/
├── README.es.md              # Spanish version (overview in Spanish)
├── README.md                 # This file (overview in English)
├── 1year/
│   ├── dataclean_1year.csv
│   ├── habitacion.csv
│   ├── autofold.py           # Automatic optimal folds detector (1 year)
│   ├── predictionsRange.py   # Range inference with 1D CNN + metrics and plots
│   ├── requirements.txt
│   ├── googleColab/
│   │   └── preprocesamientodata.ipynb
│   ├── 1D/                   # 1D CNN: notebook, final model and logs
│   ├── LSTM/                 # LSTM: notebook, final model and logs
│   └── MLP/                  # MLP: notebook, final model and logs
└── 4months/
    ├── dataclean_4months.csv
    ├── autofold.py           # Automatic optimal folds detector (4 months)
    ├── requirements.txt
    ├── 1D/                   # 1D CNN: notebook, model and logs
    ├── LSTM/                 # LSTM: notebook, model and logs
    └── MLP/                  # MLP: notebook, model and logs
```

Each model subfolder contains: training notebook, final `.h5` model, training and cross-validation checkpoints, and CV logs.

---

## 5. Datasets

| Feature | `4months` | `1year` |
|----------------|-----------|---------|
| File | `4months/dataclean_4months.csv` | `1year/dataclean_1year.csv` |
| Rows | 177,121 | 525,604 |
| Frequency | 1 minute | 1 minute |
| Period | ~4 months (since May 2025) | 12 months |
| Targets | `temperatureC`, `humidityporc`, `temperatureC2`, `humidityporc2` | The same 4 |
| Temporal features | day, month, year, hour, minute | `date` (ordinal), `hours`, `minutes` |
| Contrast file (1 year only) | — | `habitacion.csv` (real data to validate range predictions) |

**Cleaning and preprocessing** (`1year/googleColab/preprocesamientodata.ipynb`):

- Hybrid imputation of missing sensor values.
- Timestamp reconstruction from `date + hours + minutes`.
- Chronological sorting, gap detection and duplicate cleaning.
- Separate Min-Max normalization for outputs and temporal covariates.

---

## 6. Methodology

1. **Sequential formulation:** 60-step window (60 minutes) as input; simultaneous output of the 4 variables 60 minutes ahead.
2. **Temporal covariates:** ordinal date, hour and minute (and day/month/year in the 4-month module), normalized with `MinMaxScaler`.
3. **Strict temporal split:** 80% development (train + validation) and 20% hold-out test, without shuffling, preserving chronological order.
4. **Cross-validation with expanding window:** the train grows fold by fold and validation is always posterior. The number of folds is automatically calculated with `autofold.py`, which analyzes sampling frequency, gaps, minimum train size and desired validation window.
   - 4 months: 3 folds (Train 35,412 / Val 35,412; Train 70,824 / Val 35,412; Train 106,236 / Val 35,412).
   - 1 year: up to 12 configurable folds, with validation window of ~26 days (~1 month per fold).
5. **Composite selection score (4-month module):** `0.40×R²_norm + 0.30×(1−MAE_norm) + 0.20×(1−STD_norm) + 0.10×(1−Gap_norm)`, with eligibility criteria (min R² ≥ 0.20, max Gap ≤ 1.80, max STD ≤ 0.15).
6. **Evaluation metrics:** MAE, RMSE, Pearson correlation and coefficient of determination R² per variable, in original units.

---

## 7. Compared models

Three families implemented in TensorFlow/Keras, each with 4 variants in the 4-month study; the best variants were retrained in the 1-year study:

| Family | Best variant (4 months) | Parameters | R² CV | Final 1-year model |
|---------|--------------------------|------------|-------|-------------------|
| **MLP** | MLP_Light_Dropout (64→32→16→4, dropout 0.1) | 21,940 | 0.5053 | `1year/MLP/1yearbest_mlp_model_final.h5` |
| **LSTM** | LSTM_Original (LSTM 64 → Dense 32 → 4) | 20,132 | 0.5734 | `1year/LSTM/1yearbest_lstm_model.h5` |
| **1D CNN** | Conv1D_Original (3 conv layers + Dense 128→64→32→4) | 142,932 | **0.9435** | `1year/1D/best_conv1d_model_final.h5` |

The 1D CNN captures local patterns (abrupt changes of temperature/humidity in minute windows) with greater efficiency than the pure recurrence of the LSTM and than the MLP without temporal structure.

---

## 8. Main results

Results on hold-out test set of the **4-month** study (rigorous comparative reference):

| Model | R² test temp1 | R² test hum1 | R² test temp2 | R² test hum2 | Test MAE |
|--------|---------------|--------------|---------------|--------------|----------|
| **1D CNN** | **0.9789** | **0.9441** | **0.8669** | **0.9016** | **1.05** |
| MLP | 0.4710 | −1.8996 | 0.5775 | 0.2604 | 4.23 |
| LSTM | −0.2111 | −7.0546 | 0.5092 | −0.5752 | 6.65 |

Findings:

- **The 1D CNN widely and consistently outperforms MLP and LSTM** on the 4 variables, with MAE per variable between 0.14 and 2.29 in original units.
- MLP and LSTM present severe overfitting and negative R² in humidity, which indicates inability to generalize the hygrometric dynamics.
- The **1-year** study retains the three final trained models and adds operational inference by date range (`predictionsRange.py`), with comparison against real data, 4 plots (one per variable) and MAE, RMSE, correlation and R² report.

---

## 9. Scripts and utilities

- **`autofold.py`** (in `1year/` and `4months/`): determines the optimal number of folds for temporal CV. It detects sampling frequency, gaps, calculates available sequences (`rows − SEQ_LENGTH`), divides into development/test and evaluates constraints (validation window, minimum train and minimum sequence length). It prints fold-by-fold simulation, sensitivity table and the `N_FOLDS = ...` line ready for the main script.
- **`predictionsRange.py`** (only in `1year/`): inference with the 1D CNN over a `[start, end]` range defined by the user. It reconstructs the anchored 60-step window, normalizes temporal covariates, predicts, saves `prediction_vs_real.csv`, generates 4 plots (prediction vs. real) and calculates metrics per variable.
- **`googleColab/preprocesamientodata.ipynb`**: hybrid imputation, timestamp reconstruction and cleaning.

---

## 10. Installation

`conda` with Python 3.10 is recommended. Each module has its own `requirements.txt`.

```bash
# 1. Create and activate the environment
conda create -n mi_entorno python=3.10 -y
conda activate mi_entorno

# 2. Install dependencies (choose the module to work with)
pip install -r 1year/requirements.txt
# or
pip install -r 4months/requirements.txt
```

> Note: the `requirements.txt` were frozen from conda environments; if you only need the essentials, `tensorflow`, `numpy`, `pandas`, `scikit-learn` and `matplotlib` are enough.

---

## 11. Usage

**a) Calculate optimal folds:**

```bash
cd 1year
python autofold.py
# or
cd ../4months
python autofold.py
```

Adjust `CSV_PATH`, `SEQ_LENGTH`, `VENTANA_VAL_DIAS`, `MIN_TRAIN_RATIO` and `MAX_FOLDS` in the script header according to your dataset.

**b) Date-range inference (1-year module, 1D CNN model):**

```bash
cd 1year
python predictionsRange.py
```

The script will interactively request:

```text
Enter the start date for prediction (YYYY-MM-DD):
Enter the start hour for prediction (0-23):
Enter the start minute for prediction (0-59):
Enter the end date for prediction (YYYY-MM-DD):
Enter the end hour for prediction (0-23):
Enter the end minute for prediction (0-59):
```

Outputs: `prediction_vs_real.csv`, 4 prediction vs. real plots and MAE / RMSE / correlation / R² metrics per variable.

**c) Retrain or explore models:**

Open the notebooks in `1D/`, `LSTM/` and `MLP/` of each module (for example, `1year/1D/1DyearCrossval.ipynb`). The final `.h5` models are already included for direct inference.

---

## 12. Technical requirements

- Python 3.10
- TensorFlow / Keras 2.10, h5py (`.h5` models)
- NumPy 1.26.4, Pandas 2.2.3, scikit-learn, Matplotlib
- Jupyter for the training notebooks
- See `1year/requirements.txt` and `4months/requirements.txt` files for the full pinning.

---

## 13. Conclusions

A predictive model based on one-dimensional convolutional neural networks (Conv1D) was developed and validated for the joint estimation of indoor air temperature and humidity in the study room. The model reached an R² of 0.9481 and an MAE of 0.7997°C on the annual set, evidencing a high predictive capacity in the analyzed context.

The experimental comparison with LSTM and MLP showed that Conv1D offers the best balance between accuracy, temporal stability and generalization capacity for this environmental forecasting problem.

The use of a one-year dataset allowed to better capture the seasonal variability than the four-month set, with a reduction of the generalization Gap from 1.1581 to 0.0427 (more than 27 times) when going from 4 months to 1 year, so the length of the historical series decisively influences the forecast quality.

Relative humidity showed a more difficult behavior to predict than temperature, which confirms that both variables should not be interpreted with the same ease nor with the same level of confidence.

The use of low-cost hardware and open software tools demonstrated that it is feasible to build a local, reproducible and useful solution for residential environmental monitoring, although its extrapolation to other contexts requires additional validation. This approach provides specific evidence for a little-studied context in the literature: brick and cement dwellings in tropical-humid climate, with joint prediction of temperature and humidity.

The results support the use of the model as a support tool for preventive monitoring of the indoor environment, but they do not allow to affirm direct effects on the health of the occupants nor to replace certified clinical or instrumental measurements.

As future work, it is recommended to extend the validation to multiple dwellings, to incorporate exogenous variables and to explore higher-precision sensors and uncertainty quantification to improve the operational robustness of the system.


#### Potential improvements

- Conduct in-depth research to find optimal parameters for each architecture (MLP, CNN, LSTM)
- Perform the same for GRU
- Deploy or run inference using the same range as the training data

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)
