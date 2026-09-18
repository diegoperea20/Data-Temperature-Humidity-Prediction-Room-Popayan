# 4-Month Temperature and Humidity Forecasting

A multivariate time series forecasting project that predicts **temperature and humidity from two sensors** (4 target variables) **60 minutes ahead** using a sliding window of the past 60 minutes.

## Dataset

- **File:** `dataclean_4months.csv`
- **Size:** 177,121 rows, sampled every minute
- **Span:** ~4 months (starting May 2025)
- **Targets:** `temperatureC`, `humidityporc`, `temperatureC2`, `humidityporc2`
- **Features:** day, month, year, hour, minute

## Models Compared

Three deep learning architectures from TensorFlow/Keras, each with 4 variants:

| Family | Best Variant | Params | CV R² |
|--------|-------------|--------|-------|
| **MLP** | MLP_Light_Dropout (64→32→16→4, dropout 0.1) | 21,940 | 0.5053 |
| **LSTM** | LSTM_Original (LSTM 64 → Dense 32 → 4) | 20,132 | 0.5734 |
| **Conv1D** | Conv1D_Original (3 conv layers + Dense 128→64→32→4) | 142,932 | **0.9435** |

## Cross-Validation

Temporal expanding-window CV with 3 folds:

- Fold 1: Train 35,412 / Val 35,412
- Fold 2: Train 70,824 / Val 35,412
- Fold 3: Train 106,236 / Val 35,412

Composite score: `0.40×R2_norm + 0.30×(1−MAE_norm) + 0.20×(1−STD_norm) + 0.10×(1−Gap_norm)`

Eligibility: Min R² ≥ 0.20, Max Gap ≤ 1.80, Max STD ≤ 0.15

## Results

The **Conv1D_Original** model significantly outperformed both MLP and LSTM:

| Model | Test R² (temp1) | Test R² (hum1) | Test R² (temp2) | Test R² (hum2) | Test MAE |
|-------|----------------|----------------|----------------|----------------|----------|
| **Conv1D** | **0.9789** | **0.9441** | **0.8669** | **0.9016** | **1.05** |
| MLP | 0.4710 | −1.8996 | 0.5775 | 0.2604 | 4.23 |
| LSTM | −0.2111 | −7.0546 | 0.5092 | −0.5752 | 6.65 |

MLP and LSTM showed severe overfitting and negative R² on humidity variables. Conv1D achieved excellent performance across all four targets with MAEs of 0.14–2.29 in original units.

## Project Structure

```
4months/
├── autofold.py                     # Helper for CV fold computation
├── dataclean_4months.csv           # Dataset
├── requirements.txt                # Dependencies
├── MLP/                            # MLP notebook, model, logs
├── LSTM/                           # LSTM notebook, model, logs
└── 1D/                             # Conv1D notebook, model, logs
```

## Requirements

TensorFlow, NumPy, Pandas, scikit-learn, Matplotlib. See `requirements.txt`.


## 👨‍💻 Author / Autor

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)