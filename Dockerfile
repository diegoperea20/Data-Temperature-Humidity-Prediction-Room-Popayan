# =============================================================================
# Maestria Data 1year/4months — Miniconda (CPU por defecto, GPU opcional)
# Build CPU: docker build -t maestria:cpu .
# Build GPU: docker build --build-arg TF_PACKAGE="tensorflow==2.10.0" -t maestria:gpu .
# =============================================================================
# Etapa builder: crea el entorno conda (capa cacheable) y limpia caches.
# Etapa runtime: imagen final minima sin caches de conda/pip.

ARG TF_PACKAGE="tensorflow-cpu==2.10.0"

# ------------------------------- builder ------------------------------------
FROM continuumio/miniconda3:latest AS builder

WORKDIR /tmp

# environment.yml primero para aprovechar el cache de capas de Docker.
COPY environment.yml /tmp/environment.yml

# Genera una variante del environment con el paquete TF deseado (CPU o GPU).
# Solo se reescribe la linea de tensorflow-cpu si TF_PACKAGE es la version GPU.
ARG TF_PACKAGE
RUN apt-get update && apt-get install -y --no-install-recommends sed \
    && rm -rf /var/lib/apt/lists/* \
    && if [ "$TF_PACKAGE" != "tensorflow-cpu==2.10.0" ]; then \
         sed -i "s|- tensorflow-cpu==2.10.0|- ${TF_PACKAGE}|" /tmp/environment.yml; \
       fi \
    && cat /tmp/environment.yml \
    && conda install -n base -c conda-forge conda-libmamba-solver -y \
    && conda env create -n maestria --solver=libmamba -f /tmp/environment.yml \
    && conda clean -afy \
    && rm -rf /opt/conda/pkgs /root/.cache /tmp/*

# ------------------------------- runtime ------------------------------------
FROM continuumio/miniconda3:latest AS runtime

# Copia solo el entorno ya creado (sin caches del builder).
COPY --from=builder /opt/conda/envs/maestria /opt/conda/envs/maestria

ENV PATH=/opt/conda/envs/maestria/bin:$PATH \
    CONDA_DEFAULT_ENV=maestria \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Headless: predictionsRange.py usa matplotlib (savefig en vez de show).
    MPLBACKEND=Agg \
    # Rutas parametrizables (montadas por volumen en compose, ver docker-compose.yml).
    MODEL_PATH=/data/models/best_conv1d_model_final.h5 \
    TRAIN_DATASET_PATH=/data/dataclean_1year.csv \
    COMPARE_DATASET_PATH=/data/habitacion.csv \
    CSV_PATH=/data/dataclean_1year.csv \
    OUTPUT_DIR=/workspace/outputs

RUN useradd -m -u 1000 appuser \
    && mkdir -p /workspace/outputs /data/models /data \
    && chown -R appuser:appuser /workspace /data

WORKDIR /workspace

# Solo codigo (datos y .h5 van por volumen, no dentro de la imagen).
COPY --chown=appuser:appuser 1year/autofold.py ./autofold_1year.py
COPY --chown=appuser:appuser 4months/autofold.py ./autofold_4months.py
COPY --chown=appuser:appuser 1year/predictionsRange.py ./predictionsRange.py

USER appuser

EXPOSE 8888

# Por defecto: inferencia CLI. Jupyter se lanza sobrescribiendo el comando
# (ver servicio "jupyter" en docker-compose.yml).
CMD ["python", "predictionsRange.py"]
