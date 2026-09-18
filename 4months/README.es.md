# Predicción de Temperatura y Humedad a 4 Meses

Proyecto de forecasting multivariable de series temporales que predice **temperatura y humedad de dos sensores** (4 variables objetivo) **con 60 minutos de anticipación** usando una ventana deslizante de los últimos 60 minutos.

## Dataset

- **Archivo:** `dataclean_4months.csv`
- **Tamaño:** 177,121 filas, muestreadas cada minuto
- **Período:** ~4 meses (a partir de mayo 2025)
- **Variables objetivo:** `temperatureC`, `humidityporc`, `temperatureC2`, `humidityporc2`
- **Características:** día, mes, año, hora, minuto

## Modelos Comparados

Tres arquitecturas de deep learning con TensorFlow/Keras, cada una con 4 variantes:

| Familia | Mejor Variante | Parámetros | R² (VC) |
|---------|---------------|-----------|---------|
| **MLP** | MLP_Light_Dropout (64→32→16→4, dropout 0.1) | 21,940 | 0.5053 |
| **LSTM** | LSTM_Original (LSTM 64 → Dense 32 → 4) | 20,132 | 0.5734 |
| **Conv1D** | Conv1D_Original (3 capas conv + Dense 128→64→32→4) | 142,932 | **0.9435** |

## Validación Cruzada

Validación cruzada temporal con ventana expandida de 3 pliegues:

- Pliegue 1: Train 35,412 / Val 35,412
- Pliegue 2: Train 70,824 / Val 35,412
- Pliegue 3: Train 106,236 / Val 35,412

Puntaje compuesto: `0.40×R2_norm + 0.30×(1−MAE_norm) + 0.20×(1−STD_norm) + 0.10×(1−Gap_norm)`

Elegibilidad: R² mín ≥ 0.20, Gap máx ≤ 1.80, STD máx ≤ 0.15

## Resultados

El modelo **Conv1D_Original** superó significativamente tanto a MLP como a LSTM:

| Modelo | R² test (temp1) | R² test (hum1) | R² test (temp2) | R² test (hum2) | MAE test |
|--------|----------------|----------------|----------------|----------------|----------|
| **Conv1D** | **0.9789** | **0.9441** | **0.8669** | **0.9016** | **1.05** |
| MLP | 0.4710 | −1.8996 | 0.5775 | 0.2604 | 4.23 |
| LSTM | −0.2111 | −7.0546 | 0.5092 | −0.5752 | 6.65 |

MLP y LSTM mostraron sobreajuste severo y R² negativos en las variables de humedad. Conv1D logró un rendimiento excelente en las cuatro variables con MAEs de 0.14–2.29 en unidades originales.

## Estructura del Proyecto

```
4months/
├── autofold.py                     # Helper para calcular pliegues de VC
├── dataclean_4months.csv           # Dataset
├── requirements.txt                # Dependencias
├── MLP/                            # Notebook MLP, modelo, logs
├── LSTM/                           # Notebook LSTM, modelo, logs
└── 1D/                             # Notebook Conv1D, modelo, logs
```

## Requisitos

TensorFlow, NumPy, Pandas, scikit-learn, Matplotlib. Ver `requirements.txt`.


## 👨‍💻 Author / Autor

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)