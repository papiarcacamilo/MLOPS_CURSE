# =============================================================================
# MLOPS_CURSE | Imagen de despliegue del modelo de riesgo crediticio
# =============================================================================
#
# QUE PIDE LA ENTREGA 3
#
#   "... una imagen que contenga las librerias y el codigo para una app que
#    permita disponibilizar dicho objeto ..."
#
# Construir:  docker build -t mlops-riesgo:1.0.0 .
# Ejecutar :  docker run -p 8000:8000 mlops-riesgo:1.0.0
# Abrir    :  http://localhost:8000/docs
#
# QUE ENTRA Y QUE NO
#
# Entra el codigo de inferencia, el modelo serializado y los artefactos que
# describen como decidir. NO entran los datos de entrenamiento ni los notebooks:
# una imagen de servicio que carga la base de clientes es una fuga de datos
# esperando ocurrir, y ademas la engorda sin aportar nada al endpoint.
#
# Por eso se declara MLOPS_RAIZ: el codigo localiza la raiz del proyecto
# buscando Base_de_datos.csv, que aqui deliberadamente no existe.
# =============================================================================

FROM python:3.11-slim

LABEL org.opencontainers.image.title="MLOPS_CURSE - Riesgo crediticio"
LABEL org.opencontainers.image.description="Endpoint de prediccion por lote"
LABEL org.opencontainers.image.version="1.0.0"

# PYTHONDONTWRITEBYTECODE  evita .pyc en un sistema de archivos efimero
# PYTHONUNBUFFERED         hace que los logs salgan al momento, no por bloques
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLOPS_RAIZ=/app

WORKDIR /app

# Las dependencias van en su propia capa y ANTES del codigo. Asi un cambio en
# model_deploy.py no obliga a reinstalar pandas y scikit-learn en cada build.
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Codigo de inferencia. Solo los cuatro modulos que el endpoint importa:
# app -> model_deploy -> ft_engineering -> reglas_negocio.
COPY config.json .
COPY mlops_pipeline/reglas_negocio.py mlops_pipeline/
COPY mlops_pipeline/ft_engineering.py mlops_pipeline/
COPY mlops_pipeline/model_deploy.py   mlops_pipeline/
COPY mlops_pipeline/app.py            mlops_pipeline/

# Artefactos del modelo. `artefacto_despliegue.json` lo produce
# `python mlops_pipeline/model_deploy.py`, que debe ejecutarse antes del build.
COPY data/models/modelo_seleccionado.joblib   data/models/
COPY data/models/evaluacion.json              data/models/
COPY data/models/artefacto_despliegue.json    data/models/
COPY data/models/scorecard.csv                data/models/

# El registro del endpoint es la entrada de la Fase 5. Se deja como volumen
# para que sobreviva al contenedor: si se borra con el, no hay monitoreo.
RUN mkdir -p /app/data/monitoring
VOLUME ["/app/data/monitoring"]

# Usuario sin privilegios. Un proceso que solo lee artefactos y escribe un CSV
# no necesita root, y como root un fallo de la app se convierte en un fallo del
# host.
RUN useradd --create-home --shell /bin/bash servicio \
    && chown -R servicio:servicio /app
USER servicio

EXPOSE 8000

# La sonda usa /salud, que responde solo despues de que el modelo quedo cargado.
# Un contenedor que arranca sin modelo debe reportarse enfermo, no sano.
#
# Y no basta con que responda: hay que PARSEAR la respuesta y exigir el estado.
# Comprobar solo que devuelve 200 daria por sano un servicio que contestara
# {"estado": "degradado"}, que es justo el caso que la sonda debe atrapar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json,sys,urllib.request as u; sys.exit(0 if json.loads(u.urlopen('http://127.0.0.1:8000/salud').read())['estado'] == 'activo' else 1)"

CMD ["uvicorn", "app:app", "--app-dir", "mlops_pipeline", \
     "--host", "0.0.0.0", "--port", "8000"]
