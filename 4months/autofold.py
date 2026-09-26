"""
╔══════════════════════════════════════════════════════════════════╗
║         DETECTOR AUTOMÁTICO DE FOLDS ÓPTIMOS PARA CV            ║
║         Time Series — Conv1D / cualquier modelo secuencial       ║
║                                                                  ║
║  Formato CSV esperado:                                           ║
║  id, date, hours, minutes, var1, var2 ...                        ║
║  date = solo fecha (YYYY-MM-DD), hora y minuto en cols propias   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import math
import os

# ==============================
# ▶ CONFIGURACIÓN — edita aquí
# ==============================

#CSV_PATH         = os.getenv('CSV_PATH', 'dataclean_1year.csv')
CSV_PATH         = os.getenv('CSV_PATH', 'dataclean_4months.csv')

COL_DATE         = 'date'       # columna con la fecha (YYYY-MM-DD)
COL_HOURS        = 'hours'      # columna con la hora   (0-23)
COL_MINUTES      = 'minutes'    # columna con el minuto (0-59)

SEQ_LENGTH       = 60           # mismo SEQ_LENGTH que tu modelo Conv1D

DEV_RATIO        = 0.8          # fracción usada como desarrollo (debe coincidir con tu script)

VENTANA_VAL_DIAS = 15           # días que quieres que cubra cada fold de validación
                                # 7  → semanal | 15 → quincenal | 30 → mensual

# ======== PARÁMETROS DE RESTRICCIÓN PARA LOS FOLDS (ajusta según el caso en este 1 año) ========
#MIN_TRAIN_RATIO  = 0.08   # acepta que el 1er fold tenga ≥8% del dev (~26 días)
#MAX_FOLDS        = 12     # permite hasta 12
#VENTANA_VAL_DIAS = 26     # ~1 mes por fold de validación
#========

MIN_FOLDS        = 2
MAX_FOLDS        = 8
MIN_TRAIN_RATIO  = 0.25         # el 1er fold de train debe tener al menos este % del dev

# ==============================
# 1. CARGAR CSV
# ==============================

print("\n" + "=" * 65)
print("  ANÁLISIS DEL DATASET")
print("=" * 65)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No se encontró: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"  Archivo:        {CSV_PATH}")
print(f"  Filas totales:  {len(df):,}")
print(f"  Columnas:       {df.columns.tolist()}")

# Verificar columnas necesarias
for col in [COL_DATE, COL_HOURS, COL_MINUTES]:
    if col not in df.columns:
        raise ValueError(f"Columna '{col}' no encontrada. Disponibles: {df.columns.tolist()}")

# ==============================
# 2. CONSTRUIR TIMESTAMP COMPLETO
#    combinando date + hours + minutes
# ==============================

df[COL_DATE] = pd.to_datetime(df[COL_DATE])

df['_datetime'] = (
    df[COL_DATE]
    + pd.to_timedelta(df[COL_HOURS],   unit='h')
    + pd.to_timedelta(df[COL_MINUTES], unit='m')
)

df = df.sort_values('_datetime').reset_index(drop=True)

fecha_inicio = df['_datetime'].iloc[0]
fecha_fin    = df['_datetime'].iloc[-1]
duracion     = fecha_fin - fecha_inicio

print(f"\n  Fecha inicio:   {fecha_inicio}")
print(f"  Fecha fin:      {fecha_fin}")
print(f"  Duración total: {duracion.days} días ({duracion.days / 30:.1f} meses aprox.)")

# ==============================
# 3. DETECTAR FRECUENCIA DE MUESTREO
# ==============================

print("\n" + "=" * 65)
print("  DETECCIÓN DE FRECUENCIA DE MUESTREO")
print("=" * 65)

diffs         = df['_datetime'].diff().dropna()
mediana_diff  = diffs.median()
freq_segundos = mediana_diff.total_seconds()

if freq_segundos == 0:
    raise ValueError(
        "La frecuencia mediana es 0.\n"
        "Posibles causas:\n"
        "  1. Filas duplicadas con el mismo date+hours+minutes.\n"
        "  2. Las columnas hours/minutes tienen siempre el mismo valor.\n"
        "Revisa tu CSV con: df[df.duplicated(['date','hours','minutes'])].head()"
    )

freq_minutos     = freq_segundos / 60
freq_horas       = freq_minutos  / 60
muestras_por_dia = (24 * 3600) / freq_segundos

# Descripción legible
if freq_segundos < 60:
    freq_str = f"{freq_segundos:.0f} segundos"
elif freq_segundos < 3600:
    freq_str = f"{freq_minutos:.0f} minutos"
elif freq_segundos < 86400:
    freq_str = f"{freq_horas:.1f} horas"
else:
    freq_str = f"{freq_segundos/86400:.1f} días"

# Detectar gaps anómalos
p95_diff = diffs.quantile(0.95).total_seconds()
hay_gaps = p95_diff > freq_segundos * 3
pct_gaps = (diffs[diffs > mediana_diff * 3].count() / len(diffs)) * 100

print(f"  Frecuencia mediana:   {freq_str}")
print(f"  Muestras por día:     {muestras_por_dia:.1f}")
print(f"  Muestras por hora:    {muestras_por_dia / 24:.1f}")

if hay_gaps:
    print(f"  ⚠️  Gaps detectados ({pct_gaps:.1f}% de filas con salto > 3× mediana)")
    print(f"     Frecuencia p95 = {p95_diff/60:.1f} min vs mediana {freq_minutos:.0f} min")
else:
    print(f"  ✅ Sin gaps significativos")

# ==============================
# 4. MUESTRAS DISPONIBLES PARA DEV
# ==============================

print("\n" + "=" * 65)
print("  MUESTRAS PARA CROSS-VALIDATION")
print("=" * 65)

n_total     = len(df)
n_sequences = max(0, n_total - SEQ_LENGTH)
n_dev       = int(n_sequences * DEV_RATIO)
n_test      = n_sequences - n_dev

dias_dev    = n_dev  / muestras_por_dia
dias_test   = n_test / muestras_por_dia

print(f"  Filas CSV:            {n_total:,}")
print(f"  Secuencias totales:   {n_sequences:,}   (filas − SEQ_LENGTH={SEQ_LENGTH})")
print(f"  Dev  (train+val):     {n_dev:,}   → {dias_dev:.1f} días ({dias_dev/30:.1f} meses)")
print(f"  Test (hold-out):      {n_test:,}   → {dias_test:.1f} días ({dias_test/30:.1f} meses)")

# ==============================
# 5. CÁLCULO DE FOLDS ÓPTIMOS
# ==============================

def calcular_folds(n_dev, muestras_por_dia, ventana_val_dias,
                   seq_length, min_folds, max_folds, min_train_ratio):
    """
    Expanding-window CV temporal.
    temporal_kfold_indices divide en (n_folds+1) bloques iguales:
        fold_size = n_dev // (n_folds + 1)

    Restricciones:
        R1 — fold_size >= ventana_val_dias * muestras_por_dia
        R2 — fold_size >= min_train_ratio * n_dev   (train 1er fold)
        R3 — fold_size >= 2 * seq_length
    """
    muestras_val = ventana_val_dias * muestras_por_dia

    lim_r1 = math.floor(n_dev / muestras_val)       - 1  # ventana val
    lim_r2 = math.floor(1 / min_train_ratio)         - 1  # train mínimo
    lim_r3 = math.floor(n_dev / (2 * seq_length))    - 1  # secuencias mín

    n_raw   = min(lim_r1, lim_r2, lim_r3)
    n_final = max(min_folds, min(max_folds, n_raw))

    fold_size = n_dev // (n_final + 1)
    dias_fold = fold_size / muestras_por_dia

    return {
        'n_folds':        n_final,
        'n_raw':          n_raw,
        'lim_r1':         lim_r1,
        'lim_r2':         lim_r2,
        'lim_r3':         lim_r3,
        'fold_size':      fold_size,
        'dias_fold':      dias_fold,
        'meses_fold':     dias_fold / 30,
        'train_fold1':    fold_size,
        'train_fold_ult': fold_size * n_final,
        'dias_train1':    fold_size / muestras_por_dia,
        'dias_train_ult': (fold_size * n_final) / muestras_por_dia,
    }

res = calcular_folds(
    n_dev            = n_dev,
    muestras_por_dia = muestras_por_dia,
    ventana_val_dias = VENTANA_VAL_DIAS,
    seq_length       = SEQ_LENGTH,
    min_folds        = MIN_FOLDS,
    max_folds        = MAX_FOLDS,
    min_train_ratio  = MIN_TRAIN_RATIO,
)

# ==============================
# 6. RESULTADOS
# ==============================

print("\n" + "=" * 65)
print("  ANÁLISIS DE RESTRICCIONES")
print("=" * 65)
print(f"  R1 — Ventana val ({VENTANA_VAL_DIAS} días/fold):    máx {res['lim_r1']} folds")
print(f"  R2 — Train mínimo ({MIN_TRAIN_RATIO*100:.0f}% dev):       máx {res['lim_r2']} folds")
print(f"  R3 — Secuencias mín (2×{SEQ_LENGTH}):      máx {res['lim_r3']} folds")
print(f"\n  Antes de clampear [{MIN_FOLDS}–{MAX_FOLDS}]:    {res['n_raw']} folds")

print("\n" + "=" * 65)
print("  ✅  RESULTADO FINAL")
print("=" * 65)
print(f"\n  ┌──────────────────────────────────────────┐")
print(f"  │   N_FOLDS óptimo  =  {res['n_folds']:<3}                 │")
print(f"  └──────────────────────────────────────────┘")
print(f"\n  Muestras por fold:    {res['fold_size']:,}")
print(f"  Días por fold:        {res['dias_fold']:.1f} días  ({res['meses_fold']:.1f} mes aprox.)")
print(f"\n  Train del fold 1:     {res['train_fold1']:,} muestras  ({res['dias_train1']:.1f} días)")
print(f"  Train del fold {res['n_folds']}:     {res['train_fold_ult']:,} muestras  ({res['dias_train_ult']:.1f} días)")

# ==============================
# 7. SIMULACIÓN DE TODOS LOS FOLDS
# ==============================

print("\n" + "=" * 65)
print("  SIMULACIÓN — DISTRIBUCIÓN DE CADA FOLD")
print("=" * 65)
print(f"  {'Fold':<6} {'Train (muestras)':<22} {'Val (muestras)':<18} {'Val (días)'}")
print(f"  {'─'*58}")

fold_size = res['fold_size']
n_folds   = res['n_folds']

for i in range(n_folds):
    train_end = fold_size * (i + 1)
    val_start = train_end
    val_end   = min(val_start + fold_size, n_dev)
    val_size  = val_end - val_start
    dias_val  = val_size / muestras_por_dia
    print(f"  {i+1:<6} {train_end:<22,} {val_size:<18,} {dias_val:.1f} días")

# ==============================
# 8. TABLA DE SENSIBILIDAD
# ==============================

print("\n" + "=" * 65)
print("  TABLA DE SENSIBILIDAD — distintas ventanas de validación")
print("=" * 65)
print(f"  {'Ventana val':<16} {'N_FOLDS':<10} {'Días/fold':<12} {'Meses/fold'}")
print(f"  {'─'*52}")

for dias in [7, 14, 15, 21, 30, 45, 60]:
    r = calcular_folds(
        n_dev=n_dev, muestras_por_dia=muestras_por_dia,
        ventana_val_dias=dias, seq_length=SEQ_LENGTH,
        min_folds=MIN_FOLDS, max_folds=MAX_FOLDS,
        min_train_ratio=MIN_TRAIN_RATIO,
    )
    marker = " ◄ seleccionado" if dias == VENTANA_VAL_DIAS else ""
    print(f"  {dias} días{'':<10} {r['n_folds']:<10} {r['dias_fold']:<12.1f} {r['meses_fold']:.2f}{marker}")

# ==============================
# 9. LÍNEA LISTA PARA COPIAR
# ==============================

print("\n" + "=" * 65)
print("  📋 COPIA ESTA LÍNEA EN TU SCRIPT PRINCIPAL:")
print("=" * 65)
print(f"\n  N_FOLDS = {res['n_folds']}   "
      f"# auto: {res['dias_fold']:.1f} días/fold | "
      f"dataset {duracion.days} días | freq {freq_str}\n")