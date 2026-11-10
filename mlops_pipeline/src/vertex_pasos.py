"""
Entregable 5, Fase 2 - Logica de las etapas del pipeline propio
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

Cada funcion es una etapa de vertex_pipeline.py. Viven aqui y no dentro de los
componentes de Kubeflow para probarlas en local y en el CI, sin Google Cloud.
Los componentes solo agregan este directorio al path y las llaman.

    cargar_datos   BigQuery -> train y test, con el formato de cargar_particion
    entrenar       clona la receta del modelo final y la ajusta con el train
    evaluar        summarize_classification al 20,9% de rechazo, como en la Tarea 3
    registrar      sube el modelo al Model Registry con la etiqueta del commit
================================================================================
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

import model_training as mt
import vertex_datos as vd
import vertex_evaluacion as ve

TARGET = mt.fe.TARGET
NOMBRE_MODELO = "model.joblib"
# El contenedor de sklearn de Vertex AI exige ese nombre y una imagen. El modelo
# no se sirve desde ahi: usa clases propias y lo sirve la imagen de FastAPI. El
# registro es un catalogo de versiones.
IMAGEN_REGISTRO = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"


def _separador(raiz: Path) -> str:
    with open(Path(raiz) / "config.json", encoding="utf-8") as f:
        return json.load(f)["data"]["separator"]


def cargar_datos(tabla: str, region: str, ruta_train: str, ruta_test: str,
                 raiz: Path = vd.RAIZ, cliente=None) -> dict:
    """Separa la tabla de BigQuery en train y test.

    Pasa por CSV para que los tipos queden como en cargar_particion: con ese
    camino, reentrenar dio el mismo AUC-PR que el modelo publicado. El test
    conserva `id_fila` para rastrear cada prediccion; el train no lo necesita.
    """
    if cliente is None:
        from google.cloud import bigquery
        cliente = bigquery.Client(project=tabla.split(".")[0])
    tabla_bq = cliente.query(f"SELECT * FROM `{tabla}` ORDER BY {vd.ID}",
                             location=region).to_dataframe()
    sep = _separador(raiz)
    train = tabla_bq[tabla_bq[vd.PARTICION] == "TRAIN"].drop(columns=[vd.ID, vd.PARTICION])
    test = tabla_bq[tabla_bq[vd.PARTICION] == "TEST"].drop(columns=[vd.PARTICION])
    if len(train) + len(test) != len(tabla_bq):
        raise ValueError("hay filas fuera de TRAIN y TEST")
    for df, ruta in ((train, ruta_train), (test, ruta_test)):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(io.StringIO(df.to_csv(sep=sep, index=False)), sep=sep).to_csv(
            ruta, sep=sep, index=False)
    return {"train": len(train), "test": len(test),
            "moras_test": int((1 - test[TARGET]).sum())}


def entrenar(ruta_train: str, carpeta_modelo: str, raiz: Path = vd.RAIZ) -> dict:
    """Ajusta la misma receta del modelo final, sin su ajuste anterior.

    `clone` copia los hiperparametros y descarta lo aprendido, como hace
    evaluacion_final en model_evaluation.py.
    """
    receta = clone(joblib.load(Path(raiz) / "data" / "models" / "modelo_seleccionado.joblib"))
    train = pd.read_csv(ruta_train, sep=_separador(raiz))
    y = (1 - train[TARGET]).to_numpy()
    receta.fit(train.drop(columns=[TARGET]), y)
    Path(carpeta_modelo).mkdir(parents=True, exist_ok=True)
    joblib.dump(receta, Path(carpeta_modelo) / NOMBRE_MODELO)
    return {"creditos": int(len(y)), "moras": int(y.sum())}


def evaluar(carpeta_modelo: str, ruta_test: str, ruta_predicciones: str,
            raiz: Path = vd.RAIZ) -> dict:
    """Mide el modelo sobre el test con la misma funcion de la Tarea 3."""
    modelo = joblib.load(Path(carpeta_modelo) / NOMBRE_MODELO)
    test = pd.read_csv(ruta_test, sep=_separador(raiz))
    ids = test.pop(vd.ID)
    y = (1 - test[TARGET]).to_numpy()
    p = modelo.predict_proba(test.drop(columns=[TARGET]))[:, 1]
    Path(ruta_predicciones).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({vd.ID: ids, vd.OBJETIVO: y, "p_pipeline": p}).to_csv(
        ruta_predicciones, index=False)
    return ve.medir(y, p)


def registrar(uri_modelo: str, proyecto: str, region: str, nombre: str,
              etiquetas: dict, aiplatform=None) -> str:
    """Sube el modelo al Model Registry y devuelve su nombre de recurso."""
    if aiplatform is None:
        from google.cloud import aiplatform
    aiplatform.init(project=proyecto, location=region)
    modelo = aiplatform.Model.upload(
        display_name=nombre, artifact_uri=uri_modelo,
        serving_container_image_uri=IMAGEN_REGISTRO, labels=etiquetas)
    return modelo.resource_name
