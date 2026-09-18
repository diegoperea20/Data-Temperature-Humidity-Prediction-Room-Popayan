# Predicción de Series Temporales Ambientales en Interiores (1 Año)

Modelos de deep learning para predecir temperatura y humedad interior desde dos sensores, entrenados con un año de datos a nivel de minuto.

## Dataset

- **Archivo**: `dataclean_1year.csv` — 525,604 filas a frecuencia de 1 minuto
- **Features**: `date`, `hours`, `minutes`
- **Targets**: `temperatureC`, `humidityporc`, `temperatureC2`, `humidityporc2`
- **Fuente**: Dos sensores en una habitación (sensor 1: temperatureC/humidityporc, sensor 2: temperatureC2/humidityporc2)

## Modelos

| Modelo | Directorio | Archivo |
|--------|------------|---------|
| CNN 1D | `1D/` | `best_conv1d_model_final.h5` |
| LSTM | `LSTM/` | `1yearbest_lstm_model.h5` |
| MLP | `MLP/` | `1yearbest_mlp_model_final.h5` |

Cada directorio de modelo contiene checkpoints de entrenamiento, logs de validación cruzada, y checkpoints de CV.

## Scripts

- **`autofold.py`** — Detección automática de folds óptimos para validación cruzada temporal con ventana expansiva. Analiza la frecuencia del dataset, detecta gaps, y calcula el número máximo de folds respetando restricciones de train mínimo y ventana de validación.

- **`predictionsRange.py`** — Ejecuta inferencia en un rango de fechas definido por el usuario usando el modelo CNN 1D. Compara predicciones contra datos reales del sensor (`habitacion.csv`), genera 4 gráficos separados (uno por variable), e imprime métricas MAE, RMSE, correlación y R².

## Preprocesamiento

`googleColab/preprocesamientodata.ipynb` — Imputación híbrida para valores faltantes de sensores, reconstrucción de timestamps, y limpieza de datos.

## Requisitos

Ver `requirements.txt`. Construido con TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib.


## 👨‍💻 Author / Autor

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)