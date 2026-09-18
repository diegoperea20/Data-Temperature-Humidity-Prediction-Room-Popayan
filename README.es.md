# Datos de Predicción de Temperatura y Humedad Interior en Habitación Popayán

Sistema de pronóstico multivariado de series temporales para **temperatura y humedad interior** medidas por dos sensores en una habitación en Popayán, Colombia. El proyecto compara arquitecturas de aprendizaje profundo (**MLP, LSTM y CNN 1D**) en dos horizontes de datos: **4 meses** y **1 año**, con muestreo cada minuto y predicción a 60 minutos.

> 📄 Versión en inglés: [README.md](README.md) | 📁 Módulos: [`1year/`](1year/) · [`4months/`](4months/)


##### Evaluación de configuraciones de arquitecturas evaluadas en conjunto de datos de 4 meses 


| Arquitectura | Score | R² promedio | STD(R²) | MAE promedio | Gap promedio | Estado |
|---|---:|---:|---:|---:|---:|---|
| Conv1D | 0.9000 | 0.9435 | 0.0168 | 0.8608 | 0.2483 | Seleccionada |
| LSTM | -999 | 0.5734 | 0.3258 | 2.7277 | 1.3443 | Descartada |
| MLP | -999 | 0.5053 | 0.3932 | 2.9473 | 1.3689 | Descartada |






##### Evaluación de configuraciones de arquitecturas evaluadas en conjunto de datos de 1 año


| Arquitectura | Score | R² promedio | STD(R²) | MAE promedio | Gap promedio | Estado |
|---|---:|---:|---:|---:|---:|---|
| Conv1D (Conv1D_Original) | 0.9000 | 0.9481 | 0.0235 | 0.7997 | 0.2231 | Seleccionada |
| LSTM (LSTM_Original) | -999 | 0.3079 | 0.4864 | 3.2043 | 2.1986 | Descartada |
| MLP (MLP_Light_Dropout) | -999 | 0.2164 | 0.7123 | 3.5066 | 0.5221 | Descartada |


#### Comparación de arquitectura seleccionadas de conjunto de datos de 4 meses con 1 año 

| Metrica | CONV1D 4 MESES | CONV1D 1 AÑO | 
|---|---:|---:|
| R² promedio | 0.9435 | 0.9481 | 
| MAE| 0.8608 | 0.7997 | 
| GAP | 0.2483 | 0.2231 | 
| GAP FINAL| 1.1581 | 0.0427 |


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



## Repositorios de adquision de dataset y aplicacion web 

[ESP32-DHT11-to-Supabase](https://github.com/diegoperea20/ESP32-DHT11-to-Supabase)

[Web Temperature-Humidity-Prediction-Room-Popayan](https://github.com/diegoperea20/Temperature-Humidity-Prediction-Room-Popayan)



---

## 📑 Tabla de contenido

- [Datos de Predicción de Temperatura y Humedad Interior en Habitación Popayán](#datos-de-predicción-de-temperatura-y-humedad-interior-en-habitación-popayán)
        - [](#)
    - [Dataset](#dataset)
  - [📑 Tabla de contenido](#-tabla-de-contenido)
  - [1. Descripción del proyecto](#1-descripción-del-proyecto)
  - [3. Objetivos](#3-objetivos)
  - [4. Estructura general del repositorio](#4-estructura-general-del-repositorio)
  - [5. Datasets](#5-datasets)
  - [6. Metodología](#6-metodología)
  - [7. Modelos comparados](#7-modelos-comparados)
  - [8. Resultados principales](#8-resultados-principales)
  - [9. Scripts y utilidades](#9-scripts-y-utilidades)
  - [10. Instalación](#10-instalación)
  - [11. Uso](#11-uso)
  - [12. Requisitos técnicos](#12-requisitos-técnicos)
  - [13. Conclusiones](#13-conclusiones)
  - [14. Trabajo futuro](#14-trabajo-futuro)
    - [📄 License](#-license)
  - [👨‍💻 Author / Autor](#-author--autor)

---

## 1. Descripción del proyecto

Este repositorio contiene el desarrollo completo de un sistema de predicción ambiental en interiores, en el marco de formación de maestría en ciencia de datos. A partir de registros minuto a minuto de **dos sensores** (temperatura en °C y humedad relativa en %), se entrenan y evalúan modelos neuronales capaces de anticipar el comportamiento térmico y de humedad de una habitación.

El trabajo se organiza en dos estudios complementarios:

| Módulo | Dataset | Cobertura temporal | Propósito |
|--------|---------|-------------------|-----------|
| [`4months/`](4months/) | `dataclean_4months.csv` | ~4 meses desde mayo de 2025, 177.121 registros | Estudio controlado y comparativa rigurosa MLP vs. LSTM vs. CNN 1D |
| [`1year/`](1year/) | `dataclean_1year.csv` | 1 año completo, 525.604 registros | Estudio de largo plazo: estacionalidad, robustez y despliegue de inferencia por rangos de fechas |

Ambos módulos comparten la misma formulación: **ventana deslizante de 60 minutos de historia → pronóstico a 60 minutos** para las 4 variables objetivo.

**Variables objetivo (targets):**

- `temperatureC` / `humidityporc` — sensor 1
- `temperatureC2` / `humidityporc2` — sensor 2

Ver detalle en [`1year/README.es.md`](1year/README.es.md) y [`4months/README.es.md`](4months/README.es.md).


---

## 3. Objetivos

**Objetivo general:**

Desarrollar un modelo predictivo basado en aprendizaje automático para estimar la temperatura y la humedad del aire en un cuarto habitacional de vivienda en contextos climáticos húmedos, con el fin de anticipar condiciones de confort y prevenir riesgos para la salud.

**Objetivos específicos:**

• Caracterizar el comportamiento de humedad y temperatura de la habitación de estudio mediante la construcción y análisis de un conjunto de datos de series temporales.

• Seleccionar un conjunto de modelos candidatos, basados en una revisión del estado del arte que sean pertinentes para el problema de estimación conjunta de variables ambientales.

• Implementar un prototipo de interfaz de visualización interactiva que presente las predicciones generadas por los modelos implementados.

• Evaluar comparativamente el rendimiento de los modelos seleccionados, utilizando un protocolo de validación y métricas cuantitativas adecuadas para problemas de regresión, con el fin de determinar la arquitectura con el mejor balance entre precisión y eficiencia.


---

## 4. Estructura general del repositorio

```text
maestriadata1year4mo/
├── README.es.md              # Este archivo (visión general en español)
├── README.md                 # Versión en inglés
├── 1year/
│   ├── dataclean_1year.csv
│   ├── habitacion.csv
│   ├── autofold.py           # Detector automático de folds óptimos (1 año)
│   ├── predictionsRange.py   # Inferencia por rango con CNN 1D + métricas y gráficas
│   ├── requirements.txt
│   ├── googleColab/
│   │   └── preprocesamientodata.ipynb
│   ├── 1D/                   # CNN 1D: notebook, modelo final y logs
│   ├── LSTM/                 # LSTM: notebook, modelo final y logs
│   └── MLP/                  # MLP: notebook, modelo final y logs
└── 4months/
    ├── dataclean_4months.csv
    ├── autofold.py           # Detector automático de folds óptimos (4 meses)
    ├── requirements.txt
    ├── 1D/                   # CNN 1D: notebook, modelo y logs
    ├── LSTM/                 # LSTM: notebook, modelo y logs
    └── MLP/                  # MLP: notebook, modelo y logs
```

Cada subcarpeta de modelo contiene: notebook de entrenamiento, modelo `.h5` final, checkpoints de entrenamiento y de validación cruzada, y logs de CV.

---

## 5. Datasets

| Característica | `4months` | `1year` |
|----------------|-----------|---------|
| Archivo | `4months/dataclean_4months.csv` | `1year/dataclean_1year.csv` |
| Filas | 177.121 | 525.604 |
| Frecuencia | 1 minuto | 1 minuto |
| Periodo | ~4 meses (desde mayo de 2025) | 12 meses |
| Targets | `temperatureC`, `humidityporc`, `temperatureC2`, `humidityporc2` | Los mismos 4 |
| Features temporales | día, mes, año, hora, minuto | `date` (ordinal), `hours`, `minutes` |
| Archivo de contraste (solo 1 año) | — | `habitacion.csv` (datos reales para validar predicciones por rango) |

**Limpieza y preprocesamiento** (`1year/googleColab/preprocesamientodata.ipynb`):

- Imputación híbrida de valores faltantes de sensores.
- Reconstrucción de timestamps a partir de `date + hours + minutes`.
- Ordenamiento cronológico, detección de gaps y limpieza de duplicados.
- Normalización Min-Max separada para salidas y covariables temporales.

---

## 6. Metodología

1. **Formulación secuencial:** ventana de 60 pasos (60 minutos) como entrada; salida simultánea de las 4 variables a 60 minutos adelante.
2. **Covariables temporales:** fecha ordinal, hora y minuto (y día/mes/año en el módulo de 4 meses), normalizadas con `MinMaxScaler`.
3. **Partición temporal estricta:** 80 % desarrollo (train + validación) y 20 % test hold-out, sin barajado, preservando el orden cronológico.
4. **Validación cruzada con ventana expansiva:** el train crece fold a fold y la validación siempre es posterior. El número de folds se calcula automáticamente con `autofold.py`, que analiza frecuencia de muestreo, gaps, tamaño mínimo de train y ventana de validación deseada.
   - 4 meses: 3 folds (Train 35.412 / Val 35.412; Train 70.824 / Val 35.412; Train 106.236 / Val 35.412).
   - 1 año: hasta 12 folds configurables, con ventana de validación de ~26 días (~1 mes por fold).
5. **Puntaje compuesto de selección (módulo 4 meses):** `0.40×R²_norm + 0.30×(1−MAE_norm) + 0.20×(1−STD_norm) + 0.10×(1−Gap_norm)`, con criterios de elegibilidad (R² mín ≥ 0.20, Gap máx ≤ 1.80, STD máx ≤ 0.15).
6. **Métricas de evaluación:** MAE, RMSE, correlación de Pearson y coeficiente de determinación R² por variable, en unidades originales.

---

## 7. Modelos comparados

Tres familias implementadas en TensorFlow/Keras, cada una con 4 variantes en el estudio de 4 meses; las mejores variantes se reentrenaron en el estudio de 1 año:

| Familia | Mejor variante (4 meses) | Parámetros | R² CV | Modelo final 1 año |
|---------|--------------------------|------------|-------|-------------------|
| **MLP** | MLP_Light_Dropout (64→32→16→4, dropout 0.1) | 21.940 | 0.5053 | `1year/MLP/1yearbest_mlp_model_final.h5` |
| **LSTM** | LSTM_Original (LSTM 64 → Dense 32 → 4) | 20.132 | 0.5734 | `1year/LSTM/1yearbest_lstm_model.h5` |
| **CNN 1D** | Conv1D_Original (3 capas conv + Dense 128→64→32→4) | 142.932 | **0.9435** | `1year/1D/best_conv1d_model_final.h5` |

La CNN 1D captura patrones locales (cambios bruscos de temperatura/humedad en ventanas de minutos) con mayor eficiencia que la recurrencia pura del LSTM y que el MLP sin estructura temporal.

---

## 8. Resultados principales

Resultados en test hold-out del estudio de **4 meses** (referencia comparativa rigurosa):

| Modelo | R² test temp1 | R² test hum1 | R² test temp2 | R² test hum2 | MAE test |
|--------|---------------|--------------|---------------|--------------|----------|
| **CNN 1D** | **0.9789** | **0.9441** | **0.8669** | **0.9016** | **1.05** |
| MLP | 0.4710 | −1.8996 | 0.5775 | 0.2604 | 4.23 |
| LSTM | −0.2111 | −7.0546 | 0.5092 | −0.5752 | 6.65 |

Hallazgos:

- **La CNN 1D supera amplia y consistentemente a MLP y LSTM** en las 4 variables, con MAE por variable entre 0.14 y 2.29 en unidades originales.
- MLP y LSTM presentan sobreajuste severo y R² negativos en humedad, lo que indica incapacidad para generalizar la dinámica higrométrica.
- El estudio de **1 año** conserva los tres modelos finales entrenados y añade inferencia operativa por rango de fechas (`predictionsRange.py`), con comparación contra datos reales, 4 gráficas (una por variable) y reporte de MAE, RMSE, correlación y R².

---

## 9. Scripts y utilidades

- **`autofold.py`** (en `1year/` y `4months/`): determina el número óptimo de folds para CV temporal. Detecta frecuencia de muestreo, gaps, calcula secuencias disponibles (`filas − SEQ_LENGTH`), divide en desarrollo/test y evalúa restricciones (ventana de validación, train mínimo y longitud mínima de secuencia). Imprime simulación fold por fold, tabla de sensibilidad y la línea `N_FOLDS = ...` lista para el script principal.
- **`predictionsRange.py`** (solo en `1year/`): inferencia con la CNN 1D sobre un rango `[inicio, fin]` definido por el usuario. Reconstruye la ventana anclada de 60 pasos, normaliza covariables temporales, predice, guarda `prediction_vs_real.csv`, genera 4 gráficas (predicción vs. real) y calcula métricas por variable.
- **`googleColab/preprocesamientodata.ipynb`**: imputación híbrida, reconstrucción de timestamps y limpieza.

---

## 10. Instalación

Se recomienda `conda` con Python 3.10. Cada módulo tiene su propio `requirements.txt`.

```bash
# 1. Crear y activar el entorno
conda create -n mi_entorno python=3.10 -y
conda activate mi_entorno

# 2. Instalar dependencias (elegir el módulo a trabajar)
pip install -r 1year/requirements.txt
# o
pip install -r 4months/requirements.txt
```

> Nota: los `requirements.txt` fueron congelados desde entornos conda; si solo necesita lo esencial, basta con `tensorflow`, `numpy`, `pandas`, `scikit-learn` y `matplotlib`.

---

## 11. Uso

**a) Calcular folds óptimos:**

```bash
cd 1year
python autofold.py
# o
cd ../4months
python autofold.py
```

Ajuste `CSV_PATH`, `SEQ_LENGTH`, `VENTANA_VAL_DIAS`, `MIN_TRAIN_RATIO` y `MAX_FOLDS` en la cabecera del script según su dataset.

**b) Inferencia por rango de fechas (módulo 1 año, modelo CNN 1D):**

```bash
cd 1year
python predictionsRange.py
```

El script solicitará interactivamente:

```text
Enter the start date for prediction (YYYY-MM-DD):
Enter the start hour for prediction (0-23):
Enter the start minute for prediction (0-59):
Enter the end date for prediction (YYYY-MM-DD):
Enter the end hour for prediction (0-23):
Enter the end minute for prediction (0-59):
```

Salidas: `prediction_vs_real.csv`, 4 gráficas predicción vs. real y métricas MAE / RMSE / correlación / R² por variable.

**c) Reentrenar o explorar modelos:**

Abrir los notebooks en `1D/`, `LSTM/` y `MLP/` de cada módulo (por ejemplo, `1year/1D/1DyearCrossval.ipynb`). Los modelos finales `.h5` ya están incluidos para inferencia directa.

---

## 12. Requisitos técnicos

- Python 3.10
- TensorFlow / Keras 2.10, h5py (modelos `.h5`)
- NumPy 1.26.4, Pandas 2.2.3, scikit-learn, Matplotlib
- Jupyter para los notebooks de entrenamiento
- Ver archivos `1year/requirements.txt` y `4months/requirements.txt` para el pinning completo.

---

## 13. Conclusiones

Se desarrolló y validó un modelo predictivo basado en redes neuronales convolucionales unidimensionales (Conv1D) para la estimación conjunta de temperatura y humedad del aire interior en la habitación de estudio. El modelo alcanzó un R² de 0.9481 y un MAE de 0.7997°C en el conjunto anual, evidenciando una capacidad predictiva alta en el contexto analizado.

La comparación experimental con LSTM y MLP mostró que Conv1D ofrece el mejor balance entre precisión, estabilidad temporal y capacidad de generalización para este problema de forecasting ambiental.

El uso de un conjunto de datos de un año permitió capturar mejor la variabilidad estacional que el conjunto de cuatro meses, con una reducción del Gap de generalización de 1.1581 a 0.0427 (más de 27 veces) al pasar de 4 meses a 1 año, por lo que la longitud de la serie histórica influye de manera decisiva en la calidad del pronóstico.

La humedad relativa mostró un comportamiento más difícil de predecir que la temperatura, lo que confirma que ambas variables no deben interpretarse con la misma facilidad ni con el mismo nivel de confianza.

El empleo de hardware de bajo costo y herramientas de software abierto demostró que es factible construir una solución local, reproducible y útil para monitoreo ambiental residencial, aunque su extrapolación a otros contextos requiere validación adicional. Este enfoque aporta evidencia específica para un contexto poco estudiado en la literatura: viviendas de ladrillo y cemento en clima tropical-húmedo, con predicción conjunta de temperatura y humedad.

Los resultados respaldan el uso del modelo como herramienta de apoyo para monitoreo preventivo del ambiente interior, pero no permiten afirmar efectos directos sobre la salud de los ocupantes ni reemplazar mediciones clínicas o instrumentales certificadas.

Como trabajo futuro, se recomienda extender la validación a múltiples viviendas, incorporar variables exógenas y explorar sensores de mayor precisión y cuantificación de incertidumbre para mejorar la robustez operacional del sistema.




---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author / Autor

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)
