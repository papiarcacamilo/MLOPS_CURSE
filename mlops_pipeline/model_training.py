"""
Entrenamiento y seleccion de modelos
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/processed/*_train.csv y *_train_features.csv
Salida   : metricas por modelo y, al cerrar la etapa 5, el mejor modelo

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Se entrenan y evaluan diferentes modelos. De este debe resultar el objeto
     del modelo seleccionado como el mejor (model performance, consistency,
     scalability). Se deben utilizar las funciones: summarize_classification y
     build_model. Utilizar graficos comparativos para los modelos principales.
     Tabla resumen."

Los dos nombres de funcion son obligatorios. Los tres criterios de seleccion
vienen del diagrama de Venn del enunciado y se calculan aqui como columnas
explicitas, no como una mencion de paso:

    performance   AUC-PR sobre validacion cruzada
    consistency   desviacion entre folds; un modelo inestable no es desplegable
    scalability   tiempo de ajuste y de inferencia

ESTADO

    Etapa 3  Logistica sobre WoE, con revision de signos   IMPLEMENTADA
    Etapa 4  Arboles y boosting                            (pendiente)
    Etapa 5  Tratamiento del desbalance                    (pendiente)

LA REGLA QUE NO SE ROMPE

El conjunto de prueba NO se toca aqui. Toda la seleccion se hace con validacion
cruzada sobre train, con StratifiedKFold(5, shuffle=True, random_state=42), los
mismos folds que uso la sonda de la Fase 2 para que las cifras sean comparables.
El test se abre una sola vez, en model_evaluation.py.

QUE HAY QUE SUPERAR

    piso heuristico     AUC-PR 0.0753 estratificado / 0.0862 temporal
    sonda 17 features   AUC-PR 0.1468 / 0.1718

El piso dice si el modelo aporta sobre no tener modelo. La sonda dice si
aprovecha la informacion que las caracteristicas ya contienen.
================================================================================
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    _DIR_MODULO = str(Path(__file__).resolve().parent)
except NameError:
    _DIR_MODULO = str(Path.cwd().resolve())
if _DIR_MODULO not in sys.path:
    sys.path.insert(0, _DIR_MODULO)

import ft_engineering as fe  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("entrenamiento")

RUTA_RAIZ = fe.RUTA_RAIZ
RUTA_DATOS = fe.RUTA_SALIDA
RUTA_MODELOS = fe.RUTA_MODELOS
TARGET = fe.TARGET
SEMILLA = fe.SEMILLA

# Punto de operacion del baseline heuristico: rechaza el 20.9% de las
# solicitudes. Se reutiliza como k para el recall@k, de modo que la comparacion
# entre modelo y regla se haga sobre el mismo volumen de rechazos.
K_OPERATIVO = 0.209


def folds_estandar() -> StratifiedKFold:
    """Los mismos folds en todo el proyecto. Sin esto nada es comparable."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


# ==============================================================================
# METRICAS
# ==============================================================================


def summarize_classification(y_verdadero, y_prob, umbral: float = 0.5,
                             k: float | None = None) -> dict:
    """Resume el desempenio de un clasificador binario desbalanceado.

    Recibe PROBABILIDADES, no etiquetas, porque con 4.75% de eventos la mayor
    parte de la informacion esta en el ORDEN que el modelo impone, no en la
    decision que toma a un umbral arbitrario.

    Metricas de ordenacion, independientes del umbral:

        auc_pr    PRINCIPAL. Con esta tasa de eventos es la que refleja el
                  desempenio real. La exactitud queda descartada: un modelo que
                  apruebe todo acierta el 95.25% sin aportar nada
        ks        separacion maxima entre las distribuciones acumuladas de
                  buenos y malos. Por encima de 0.30 se considera utilizable
        gini      2*AUC_ROC-1, estandar regulatorio comparable entre entidades

    Metrica de magnitud, no de orden:

        brier     error cuadratico medio de las probabilidades. Discriminacion y
                  calibracion son propiedades independientes: un modelo puede
                  ordenar perfecto y estar pesimo calibrado, y es esperable con
                  class_weight="balanced", que distorsiona las probabilidades por
                  disenio

    Metricas de decision, en el punto de operacion:

        precision, recall y f1 al umbral dado, mas la version @k, que ordena por
        probabilidad y toma el k por uno superior. El recall@k traduce el modelo
        a una decision de negocio: si se rechaza al 20.9% peor, cuanta mora se
        evita. Es lo unico directamente comparable contra la regla heuristica.
    """
    y = np.asarray(y_verdadero).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_verdadero {y.shape} y y_prob {p.shape} no coinciden")

    eventos = int(y.sum())
    tasa_base = float(y.mean())

    auc_pr = float(average_precision_score(y, p))
    auc_roc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    ks = float(np.max(tpr - fpr))

    pred = (p >= umbral).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)

    resumen = {
        "n": int(len(y)),
        "eventos": eventos,
        "tasa_base": round(tasa_base, 4),
        "auc_pr": round(auc_pr, 4),
        "lift_vs_azar": round(auc_pr / tasa_base, 2) if tasa_base else None,
        "ks": round(ks, 4),
        "gini": round(2 * auc_roc - 1, 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "umbral": umbral,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pct_marcado": round(float(pred.mean()) * 100, 2),
    }

    if k is not None:
        n_k = max(1, int(round(len(p) * k)))
        peores = np.argsort(-p, kind="stable")[:n_k]
        resumen.update({
            "k": k,
            "recall_at_k": round(float(y[peores].sum() / eventos), 4) if eventos else None,
            "precision_at_k": round(float(y[peores].mean()), 4),
            "lift_at_k": round(float(y[peores].mean() / tasa_base), 2) if tasa_base else None,
        })

    return resumen


# ==============================================================================
# ENTRENAMIENTO
# ==============================================================================


def build_model(estimador, X, y, cv=None, nombre: str = "",
                k: float | None = K_OPERATIVO) -> dict:
    """Entrena y evalua un estimador con validacion cruzada estratificada.

    Devuelve el modelo ajustado sobre TODO el conjunto recibido mas su resumen
    de desempenio, con los tres criterios del enunciado.

    `estimador` puede ser un clasificador suelto o un Pipeline completo. Si es un
    Pipeline que incluye la ingenieria de caracteristicas, el ajuste ocurre
    dentro de cada fold y no hay fuga; si es un clasificador sobre una matriz ya
    transformada, la transformacion vio el fold de validacion. La diferencia se
    mide en este modulo, no se asume.

    Se guardan tambien las predicciones fuera de fold (`oof`), que son las que
    permiten construir despues las curvas y el analisis de calibracion sin
    volver a entrenar.
    """
    cv = folds_estandar() if cv is None else cv
    y = np.asarray(y).astype(int)

    oof = np.full(len(y), np.nan)
    por_fold, t_ajuste, t_prediccion = [], [], []

    for i_tr, i_va in cv.split(X, y):
        X_tr = X.iloc[i_tr] if hasattr(X, "iloc") else X[i_tr]
        X_va = X.iloc[i_va] if hasattr(X, "iloc") else X[i_va]

        modelo = clone(estimador)

        t0 = time.perf_counter()
        modelo.fit(X_tr, y[i_tr])
        t_ajuste.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        p_va = modelo.predict_proba(X_va)[:, 1]
        t_prediccion.append(time.perf_counter() - t0)

        oof[i_va] = p_va
        por_fold.append(summarize_classification(y[i_va], p_va, k=k))

    auc_folds = [f["auc_pr"] for f in por_fold]

    # El modelo devuelto se reajusta sobre todo el conjunto: los de cada fold
    # solo sirvieron para estimar el desempenio.
    final = clone(estimador).fit(X, y)

    return {
        "nombre": nombre or type(estimador).__name__,
        "modelo": final,
        # performance, medido sobre las predicciones fuera de fold
        "resumen_oof": summarize_classification(y, oof, k=k),
        # consistency
        "auc_pr_media": round(float(np.mean(auc_folds)), 4),
        "auc_pr_desv": round(float(np.std(auc_folds)), 4),
        "auc_pr_folds": auc_folds,
        # scalability
        "seg_ajuste": round(float(np.mean(t_ajuste)), 3),
        "seg_prediccion_por_1000": round(
            float(np.mean(t_prediccion)) / max(len(y) / len(por_fold), 1) * 1000, 4),
        "por_fold": por_fold,
        "oof": oof,
    }


# ==============================================================================
# ETAPA 3 | LOGISTICA SOBRE WoE
# ==============================================================================


def _clasificador(modelo):
    """Devuelve el estimador final, tanto si viene suelto como dentro de un Pipeline."""
    return modelo.steps[-1][1] if isinstance(modelo, Pipeline) else modelo


def revisar_signos(modelo, columnas) -> dict:
    """Comprueba que los coeficientes de las variables WoE sean positivos.

    Es la revision estandar antes de firmar un scorecard, y no es una
    formalidad. El WoE se construye de modo que un valor MAS ALTO significa MAS
    riesgo. Si su coeficiente sale negativo, el modelo esta diciendo que mas
    riesgo observado implica menos probabilidad de impago, que es imposible.
    Cuando ocurre, la causa suele ser colinealidad entre variables o un WoE mal
    orientado, y en ambos casos el modelo no es defendible ante un supervisor
    aunque su AUC-PR sea bueno.

    Solo se revisan las columnas `woe_`. Las monetarias escaladas y las banderas
    no tienen una direccion de riesgo conocida a priori, asi que su signo no
    prueba nada.
    """
    coef = _clasificador(modelo).coef_[0]
    tabla = pd.DataFrame({"variable": list(columnas), "coeficiente": coef})
    tabla["es_woe"] = tabla["variable"].str.startswith("woe_")

    woe = tabla[tabla["es_woe"]]
    invertidos = woe[woe["coeficiente"] < 0]["variable"].tolist()

    return {
        "n_woe": int(len(woe)),
        "n_invertidos": len(invertidos),
        "invertidos": invertidos,
        "todos_positivos": len(invertidos) == 0,
        "tabla": tabla.sort_values("coeficiente", ascending=False),
    }


def logistica_woe() -> Pipeline:
    """Modelo de referencia de la etapa 3.

    Regresion logistica sobre las caracteristicas WoE. Es el modelo con el que
    se comparan todos los demas, y el candidato natural a desplegarse: en credito
    la interpretabilidad no es un empate tecnico, es una ventaja, porque hay que
    justificar la negacion ante el cliente y ante el supervisor.

    class_weight="balanced" compensa el desbalance de 20:1 reponderando las
    clases. Distorsiona las probabilidades por disenio, de modo que la
    calibracion habra que revisarla en model_evaluation.py.

    El escalado va DENTRO del pipeline para que se ajuste en cada fold de
    entrenamiento y no en el de validacion.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced",
                           random_state=SEMILLA),
    )


def cargar(particion: str, con_features: bool) -> pd.DataFrame:
    sufijo = "_train_features" if con_features else "_train"
    return pd.read_csv(RUTA_DATOS / f"{particion}{sufijo}.csv",
                       sep=fe.SEPARADOR, encoding=fe.ENCODING)


def piso_heuristico() -> dict:
    """Lee el piso que fijo hueristic_model.py."""
    ruta = RUTA_MODELOS / "baseline_heuristico.json"
    if not ruta.exists():
        log.warning("No existe %s: ejecuta antes hueristic_model.py", ruta.name)
        return {}
    b = json.load(open(ruta, encoding="utf-8"))
    return {p: r["score_como_ranker"]["auc_pr"] for p, r in b["resultados"].items()}


def main() -> dict:
    log.info("=" * 72)
    log.info("ETAPA 3 | Regresion logistica sobre WoE")
    log.info("=" * 72)

    piso = piso_heuristico()
    resultados = {}

    for particion in ["estratificado", "temporal"]:
        log.info("-" * 72)
        matriz = cargar(particion, con_features=True)
        crudo = cargar(particion, con_features=False)
        X = matriz.drop(columns=[TARGET])
        y = 1 - matriz[TARGET]

        log.info("[%s] %s registros | %s caracteristicas | mora %.2f%%",
                 particion, len(X), X.shape[1], y.mean() * 100)

        # --- A) Sobre la matriz ya transformada -------------------------------
        # Es el montaje de la sonda de la Fase 2, y arrastra su misma fuga: el
        # WoE se ajusto sobre TODO train, de modo que vio los folds de
        # validacion. Se mide para poder compararlo, no porque sea correcto.
        a = build_model(logistica_woe(), X, y, nombre="logistica_woe_matriz")

        # --- B) Con la ingenieria dentro del pipeline -------------------------
        # El WoE se reajusta en cada fold de entrenamiento, de modo que el fold
        # de validacion nunca influye en la transformacion. Es el montaje
        # correcto, y solo es posible desde que la Fase 2 quedo envuelta en un
        # Pipeline de sklearn.
        completo = Pipeline([
            ("caracteristicas", fe.construir_pipeline().named_steps["caracteristicas"]),
            ("modelo", logistica_woe()),
        ])
        b = build_model(completo, crudo.drop(columns=[TARGET]), y,
                        nombre="logistica_woe_pipeline")

        fuga = round(a["auc_pr_media"] - b["auc_pr_media"], 4)

        log.info("A) sobre la matriz ya transformada  AUC-PR = %.4f +/- %.4f",
                 a["auc_pr_media"], a["auc_pr_desv"])
        log.info("B) con la ingenieria dentro del CV  AUC-PR = %.4f +/- %.4f",
                 b["auc_pr_media"], b["auc_pr_desv"])
        log.info("   optimismo atribuible a la fuga del WoE: %+.4f", fuga)

        r = b["resumen_oof"]
        log.info("Desempenio sin fuga: KS = %.4f | Gini = %.4f | Brier = %.5f",
                 r["ks"], r["gini"], r["brier"])
        log.info("En el punto de operacion del heuristico (rechazar el %.1f%%): "
                 "recall %.1f%% | precision %.2f%% (%.2fx la tasa base)",
                 K_OPERATIVO * 100, r["recall_at_k"] * 100,
                 r["precision_at_k"] * 100, r["lift_at_k"])

        if particion in piso:
            log.info("Contra el piso heuristico (%.4f): %.2fx",
                     piso[particion], b["auc_pr_media"] / piso[particion])

        # --- Revision de signos ----------------------------------------------
        signos = revisar_signos(a["modelo"], X.columns)
        if signos["todos_positivos"]:
            log.info("Revision de signos: los %s coeficientes WoE son positivos",
                     signos["n_woe"])
        else:
            log.warning("Revision de signos: %s coeficientes WoE INVERTIDOS: %s",
                        signos["n_invertidos"], signos["invertidos"])

        resultados[particion] = {
            "sobre_matriz": {k: v for k, v in a.items()
                             if k not in ("modelo", "oof")},
            "con_pipeline": {k: v for k, v in b.items()
                             if k not in ("modelo", "oof")},
            "optimismo_por_fuga": fuga,
            "signos": {k: v for k, v in signos.items() if k != "tabla"},
            "coeficientes": signos["tabla"].to_dict(orient="records"),
            "piso_heuristico": piso.get(particion),
        }

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    destino = RUTA_MODELOS / "etapa3_logistica.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({
            "etapa": "3 - Regresion logistica sobre WoE",
            "nota_test": "El conjunto de prueba NO se ha utilizado.",
            "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "resultados": resultados,
        }, f, indent=2, ensure_ascii=False)
    log.info("-" * 72)
    log.info("Guardado: %s", destino.relative_to(RUTA_RAIZ))
    log.info("=" * 72)
    return resultados


if __name__ == "__main__":
    main()
