"""
Entregable 5, Fase 2 - Python, el pipeline propio y AutoML, con la misma medicion
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : las predicciones de AutoML sobre el test, exportadas a BigQuery; las
           del pipeline propio, guardadas por su corrida en Cloud Storage; y el
           modelo servido (modelo_seleccionado.joblib)
Salida   : data/models/predicciones_test_comparacion.csv
           data/models/comparacion_vertex.json
           data/models/figuras/vertex/comparacion_modelos.png

Las tres columnas de probabilidad se miden con la misma funcion,
`summarize_classification`, sobre los mismos 2.153 creditos. El pipeline
reentrena en la nube la misma receta que Python: si sus probabilidades
coinciden, el entrenamiento es reproducible fuera del equipo.

Por que el umbral sale del propio puntaje de cada modelo: AutoML solo exporta
predicciones del test, asi que no hay un conjunto aparte donde fijarle el
umbral como se hizo con Python. Para los dos se rechaza el 20,9% de creditos de
mayor riesgo segun su puntaje. Eso no usa las etiquetas del test, asi que no hay
fuga, y deja a los dos modelos con el mismo volumen de rechazo. Las metricas de
orden (AUC-PR, ROC-AUC, KS y Brier) no dependen del umbral.

Las predicciones alineadas quedan en un CSV versionado. Asi la comparacion y el
analisis de estabilidad se repiten sin consultar Google Cloud.

Uso:
    python mlops_pipeline/src/vertex_evaluacion.py               usa el CSV versionado
    python mlops_pipeline/src/vertex_evaluacion.py --descargar   lo rehace desde la nube
================================================================================
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import model_training as mt
import vertex_datos as vd

RUTA_MODELOS = vd.RAIZ / "data" / "models"
NOMBRE_CSV = "predicciones_test_comparacion.csv"
NOMBRE_JSON = "comparacion_vertex.json"
NOMBRE_FIGURA = Path("figuras") / "vertex" / "comparacion_modelos.png"

# Variacion de AUC-PR entre pliegues: una diferencia menor es un empate.
RUIDO_AUC_PR = 0.0159
MODELOS = {"python": "Python, logistica sobre WoE",
           "pipeline": "Pipeline propio en Vertex AI",
           "automl": "AutoML Tabular"}


def leer_export_automl(config: dict, cliente=None) -> pd.DataFrame:
    """Lee la tabla que exporto AutoML y extrae la probabilidad de mora."""
    if cliente is None:
        from google.cloud import bigquery
        cliente = bigquery.Client(project=config["vertex"]["proyecto"])
    v = config["vertex"]
    tabla = f"{v['proyecto']}.{v['dataset']}.{v['automl']['tabla_predicciones']}"
    export = cliente.query(f"SELECT * FROM `{tabla}`",
                           location=v["region"]).to_dataframe()
    export["p_automl"] = export["predicted_mora"].map(
        lambda r: dict(zip(r["classes"], r["scores"]))["1"])
    return export.drop(columns=["predicted_mora"])


def alinear(test: pd.DataFrame, export: pd.DataFrame, entradas: list[str]) -> np.ndarray:
    """Devuelve la probabilidad de AutoML en el orden del test local.

    AutoML no exporta `id_fila`. Cada credito se reconoce por sus entradas y su
    objetivo, y el emparejamiento tiene que ser uno a uno.
    """
    claves = [*entradas, vd.OBJETIVO]

    def normalizar(df: pd.DataFrame) -> pd.DataFrame:
        # BigQuery exporta los enteros como texto. Una columna es numerica si
        # todos sus valores no nulos se pueden leer como numero.
        df = df.copy()
        for c in claves:
            numerica = pd.to_numeric(df[c], errors="coerce")
            es_numerica = numerica.notna().sum() == df[c].notna().sum()
            df[c] = numerica.astype(float) if es_numerica else df[c].astype(str)
        return df

    izquierda = normalizar(test[claves]).reset_index(drop=True)
    derecha = normalizar(export[[*claves, "p_automl"]])
    if izquierda.duplicated(claves).any() or derecha.duplicated(claves).any():
        raise ValueError("hay creditos repetidos; el emparejamiento no seria uno a uno")
    unido = izquierda.merge(derecha, on=claves, how="left")
    if len(unido) != len(test) or unido["p_automl"].isna().any():
        faltan = int(unido["p_automl"].isna().sum())
        raise ValueError(f"{faltan} creditos del test no tienen prediccion de AutoML")
    return unido["p_automl"].to_numpy()


def leer_predicciones_pipeline(config: dict, cliente=None, almacenamiento=None) -> pd.DataFrame:
    """Lee las predicciones que guardo la etapa `evaluar` de la corrida configurada."""
    v = config["vertex"]
    if cliente is None:
        from google.cloud import aiplatform
        cliente = aiplatform.gapic.PipelineServiceClient(
            client_options={"api_endpoint": f"{v['region']}-aiplatform.googleapis.com"})
    corrida = cliente.get_pipeline_job(name=v["pipeline"]["corrida"])
    if corrida.state.name != "PIPELINE_STATE_SUCCEEDED":
        raise ValueError(f"la corrida no termino bien: {corrida.state.name}")
    tarea = next(t for t in corrida.job_detail.task_details if t.task_name == "evaluar")
    uri = tarea.outputs["predicciones"].artifacts[0].uri
    if almacenamiento is None:
        from google.cloud import storage
        almacenamiento = storage.Client(project=v["proyecto"])
    bucket, _, nombre = uri.removeprefix("gs://").partition("/")
    datos = almacenamiento.bucket(bucket).blob(nombre).download_as_bytes()
    return pd.read_csv(io.BytesIO(datos))[[vd.ID, "p_pipeline"]]


def alinear_pipeline(ids: pd.Series, pipeline: pd.DataFrame) -> np.ndarray:
    """Probabilidad del pipeline en el orden del test local, emparejada por `id_fila`."""
    if pipeline[vd.ID].duplicated().any() or set(pipeline[vd.ID]) != set(ids):
        raise ValueError("las predicciones del pipeline no son las del test")
    return pipeline.set_index(vd.ID).loc[ids, "p_pipeline"].to_numpy()


def probabilidades_python(test: pd.DataFrame) -> np.ndarray:
    """Probabilidad de mora del modelo servido sobre el test."""
    modelo = joblib.load(RUTA_MODELOS / "modelo_seleccionado.joblib")
    return modelo.predict_proba(test.drop(columns=[mt.fe.TARGET]))[:, 1]


def verificar_python(y: np.ndarray, p: np.ndarray, ruta: Path = RUTA_MODELOS) -> None:
    """La probabilidad de Python debe reproducir el AUC-PR publicado."""
    with open(ruta / "evaluacion.json", encoding="utf-8") as f:
        publicado = json.load(f)["evaluacion_final"]["estratificado"]["metricas"]["auc_pr"]
    medido = round(float(average_precision_score(y, p)), 4)
    if medido != publicado:
        raise ValueError(f"Python no reproduce su AUC-PR publicado: {medido} contra {publicado}")


def medir(y, p, k: float = mt.K_OPERATIVO) -> dict:
    """`summarize_classification` al volumen de rechazo k, mas lo que el enunciado pide."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    umbral = float(np.quantile(p, 1 - k))
    resumen = mt.summarize_classification(y, p, umbral=umbral, k=k)
    marcado = p >= umbral
    tp = int((marcado & (y == 1)).sum())
    fp = int((marcado & (y == 0)).sum())
    fn = int((~marcado & (y == 1)).sum())
    tn = int((~marcado & (y == 0)).sum())
    resumen.update({
        "umbral": round(umbral, 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "rmse": round(float(np.sqrt(np.mean((p - y) ** 2))), 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "accuracy_trivial": round(float((y == 0).mean()), 4),
        "matriz": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    })
    return resumen


def comparar(python: dict, otro: dict, ruido: float = RUIDO_AUC_PR,
             nombre: str = "AutoML") -> dict:
    """Diferencias del otro modelo menos Python y veredicto sobre la metrica principal."""
    metricas = ("auc_pr", "roc_auc", "ks", "gini", "brier", "rmse",
                "precision", "recall", "f1", "accuracy")
    diferencias = {m: round(otro[m] - python[m], 4) for m in metricas}
    d = diferencias["auc_pr"]
    if abs(d) < ruido:
        veredicto = "empate"
    else:
        veredicto = f"gana {nombre}" if d > 0 else "gana Python"
    return {"diferencias": diferencias, "ruido_auc_pr": ruido, "veredicto_auc_pr": veredicto}


def tabla(resultado: dict) -> pd.DataFrame:
    columnas = ["auc_pr", "roc_auc", "ks", "gini", "brier", "rmse",
                "precision", "recall", "f1", "accuracy", "pct_marcado"]
    filas = {MODELOS[m]: {c: resultado[m][c] for c in columnas} for m in MODELOS}
    filas["Diferencia (pipeline - Python)"] = {
        c: resultado["comparacion_pipeline"]["diferencias"].get(c) for c in columnas}
    filas["Diferencia (AutoML - Python)"] = {
        c: resultado["comparacion"]["diferencias"].get(c) for c in columnas}
    return pd.DataFrame(filas).T


def graficar(y, probabilidades: dict, ruta: Path) -> None:
    """Curvas PR y ROC. El pipeline va punteado: si coincide con Python, se superpone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    estilos = {"python": ("#1f6fb4", "-", 2.5), "pipeline": ("#2a9d5c", ":", 2.0),
               "automl": ("#d1641b", "-", 1.5)}
    fig, (pr, roc) = plt.subplots(1, 2, figsize=(11, 4.5))
    for clave, p in probabilidades.items():
        color, linea, ancho = estilos[clave]
        precision, recall, _ = precision_recall_curve(y, p)
        pr.plot(recall, precision, color=color, linestyle=linea, linewidth=ancho,
                label=f"{MODELOS[clave]} (AUC-PR {average_precision_score(y, p):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        roc.plot(fpr, tpr, color=color, linestyle=linea, linewidth=ancho,
                 label=f"{MODELOS[clave]} (ROC-AUC {roc_auc_score(y, p):.3f})")
    pr.axhline(np.mean(y), color="grey", linestyle="--", label=f"Azar ({np.mean(y):.3f})")
    roc.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Azar")
    pr.set(title="Curva precision-recall, clase mora", xlabel="Recall", ylabel="Precision")
    roc.set(title="Curva ROC", xlabel="Tasa de falsos positivos",
            ylabel="Tasa de verdaderos positivos")
    for eje in (pr, roc):
        eje.legend(loc="best", fontsize=8)
        eje.grid(alpha=0.3)
    fig.suptitle("Mismo test estratificado: 2.153 creditos y 102 moras")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def preparar_predicciones(config: dict, entradas: list[str], export: pd.DataFrame,
                          pipeline: pd.DataFrame) -> pd.DataFrame:
    """Une la etiqueta y las tres probabilidades, en el orden del test local."""
    test_crudo = vd.leer_particion("estratificado_test", config)
    test = vd.construir_tabla(
        vd.leer_particion("estratificado_train", config), test_crudo, entradas,
        objetivo_origen=config["target_variable"], semilla=config["random_state"],
        proporcion_validacion=config["vertex"]["proporcion_validacion"])
    test = test[test[vd.PARTICION] == "TEST"].reset_index(drop=True)
    return pd.DataFrame({
        vd.ID: test[vd.ID],
        vd.OBJETIVO: test[vd.OBJETIVO],
        "p_python": probabilidades_python(test_crudo),
        "p_pipeline": alinear_pipeline(test[vd.ID], pipeline),
        "p_automl": alinear(test, export, entradas),
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--descargar", action="store_true",
                        help="rehace las predicciones desde BigQuery y Cloud Storage")
    parser.add_argument("--salida", type=Path, default=RUTA_MODELOS,
                        help="carpeta de salida; por defecto data/models")
    args = parser.parse_args(argv)

    config = vd.cargar_config()
    ruta_csv = args.salida / NOMBRE_CSV
    if args.descargar:
        predicciones = preparar_predicciones(config, vd.entradas_del_modelo(),
                                             leer_export_automl(config),
                                             leer_predicciones_pipeline(config))
        args.salida.mkdir(parents=True, exist_ok=True)
        predicciones.to_csv(ruta_csv, index=False)
    else:
        predicciones = pd.read_csv(RUTA_MODELOS / NOMBRE_CSV)

    y = predicciones[vd.OBJETIVO].to_numpy()
    if len(y) != vd.TEST_ESPERADO or int(y.sum()) != vd.MORAS_TEST_ESPERADAS:
        print("RECHAZADO: las predicciones no son del test de Python.")
        return 1
    verificar_python(y, predicciones["p_python"].to_numpy())

    resultado = {
        "test": {"creditos": int(len(y)), "moras": int(y.sum())},
        "volumen_de_rechazo": mt.K_OPERATIVO,
        "python": medir(y, predicciones["p_python"]),
        "pipeline": medir(y, predicciones["p_pipeline"]),
        "automl": medir(y, predicciones["p_automl"]),
    }
    resultado["comparacion"] = comparar(resultado["python"], resultado["automl"])
    resultado["comparacion_pipeline"] = comparar(resultado["python"], resultado["pipeline"],
                                                 nombre="el pipeline")
    resultado["reproducibilidad"] = {
        "diferencia_maxima_de_probabilidad": float(
            (predicciones["p_pipeline"] - predicciones["p_python"]).abs().max())}

    with open(args.salida / NOMBRE_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    graficar(y, {m: predicciones[f"p_{m}"] for m in MODELOS}, args.salida / NOMBRE_FIGURA)

    with pd.option_context("display.width", 200):
        print(tabla(resultado).to_string())
    print(f"Veredicto en AUC-PR, con empate por debajo de {RUIDO_AUC_PR}: "
          f"{resultado['comparacion']['veredicto_auc_pr']}")
    print(f"Pipeline frente a Python: {resultado['comparacion_pipeline']['veredicto_auc_pr']}; "
          f"diferencia maxima de probabilidad "
          f"{resultado['reproducibilidad']['diferencia_maxima_de_probabilidad']:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
