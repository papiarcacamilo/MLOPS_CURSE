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
    Etapa 4  Arboles y boosting                            IMPLEMENTADA
    Etapa 5  Tratamiento del desbalance                    IMPLEMENTADA

LA REGLA QUE NO SE ROMPE

El conjunto de prueba NO se toca aqui. Toda la seleccion se hace con validacion
cruzada sobre train, con StratifiedKFold(5, shuffle=True, random_state=42), los
mismos folds que uso la sonda de la Fase 2 para que las cifras sean comparables.
El test se abre una sola vez, en model_evaluation.py.

QUE HAY QUE SUPERAR

    piso heuristico     AUC-PR 0.0753 estratificado / 0.0862 temporal
    logistica sobre WoE AUC-PR 0.1323 / 0.1646, sin fuga, medida en la etapa 3

El piso dice si el modelo aporta sobre no tener modelo. La sonda dice si
aprovecha la informacion que las caracteristicas ya contienen.
================================================================================
"""

from __future__ import annotations

import joblib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineImb
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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


# ==============================================================================
# ETAPAS 4 Y 5 | CATALOGO DE MODELOS Y TRATAMIENTOS DEL DESBALANCE
# ==============================================================================
# Se comparan tres familias contra tres tratamientos, con la ingenieria de
# caracteristicas DENTRO de la validacion cruzada en todos los casos. La etapa 3
# midio que ajustarla fuera infla el AUC-PR en +0.0145, asi que aqui no se
# repite ese montaje.
#
# SOBRE EL AJUSTE DE UMBRAL
#
# El enunciado lo lista junto a class_weight y SMOTE, pero no es comparable en
# la misma tabla: AUC-PR es independiente del umbral, de modo que moverlo no
# cambia esa cifra en absoluto. El umbral no es una forma de entrenar, es la
# eleccion del punto de operacion sobre un modelo YA entrenado, y se ve en el
# recall y la precision @k. Se decide al final, sobre el modelo elegido.


def catalogo_modelos() -> dict:
    """Las tres familias a comparar, y por que estas.

    logistica    Es el modelo de referencia y el candidato natural a
                 desplegarse. Sobre WoE produce un scorecard: cada tramo aporta
                 unos puntos y la negacion se justifica ante el cliente y ante
                 el supervisor. En credito eso no es un empate tecnico, es una
                 ventaja.
    bosque       Captura interacciones que la logistica no ve, sin exigir que
                 la relacion sea monotona.
    boosting     HistGradientBoosting de scikit-learn, en lugar de XGBoost, para
                 no anadir una dependencia mas. Suele ser el techo de
                 rendimiento en datos tabulares.
    """
    return {
        "logistica": lambda cw: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight=cw,
                               random_state=SEMILLA)),
        "bosque": lambda cw: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight=cw,
            n_jobs=-1, random_state=SEMILLA),
        "boosting": lambda cw: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, min_samples_leaf=20,
            class_weight=cw, random_state=SEMILLA),
    }


def construir_configuracion(familia: str, tratamiento: str):
    """Arma el pipeline completo: caracteristicas, desbalance y clasificador.

    La ingenieria va siempre dentro, de modo que se reajusta en cada fold de
    entrenamiento y el fold de validacion nunca influye en la transformacion.

    Tratamientos:
        ninguno        el desbalance de 20:1 se deja como esta
        class_weight   reponderacion de clases. Distorsiona las probabilidades
                       por disenio, asi que la calibracion habra que revisarla
        smote          sobremuestreo sintetico de la clase minoritaria. Se
                       aplica SOLO al fold de entrenamiento: el Pipeline de
                       imblearn no lo ejecuta al transformar el de validacion,
                       que es justo el error que inflaria las metricas
    """
    fabrica = catalogo_modelos()[familia]
    caracteristicas = fe.construir_pipeline().named_steps["caracteristicas"]

    if tratamiento == "smote":
        # SMOTE interpola entre vecinos de la clase minoritaria. Sobre WoE eso
        # produce valores que no corresponden a ningun tramo real, de modo que
        # el cliente sintetico no es explicable. Se mide igualmente: la decision
        # se toma con evidencia, no con el argumento.
        return PipelineImb([
            ("caracteristicas", caracteristicas),
            ("smote", SMOTE(random_state=SEMILLA, k_neighbors=5)),
            ("modelo", fabrica(None)),
        ])

    cw = "balanced" if tratamiento == "class_weight" else None
    return Pipeline([
        ("caracteristicas", caracteristicas),
        ("modelo", fabrica(cw)),
    ])


def comparar_modelos(crudo: pd.DataFrame, y, familias=None,
                     tratamientos=None) -> tuple[pd.DataFrame, dict]:
    """Ejecuta la rejilla completa y devuelve la tabla resumen del enunciado.

    Las columnas reproducen los tres criterios del diagrama de Venn en lugar de
    mencionarlos: performance, consistency y scalability.
    """
    familias = familias or list(catalogo_modelos())
    tratamientos = tratamientos or ["ninguno", "class_weight", "smote"]
    X = crudo.drop(columns=[TARGET])

    filas, detalle = [], {}
    for familia in familias:
        for tratamiento in tratamientos:
            etiqueta = f"{familia}+{tratamiento}"
            log.info("   entrenando %s", etiqueta)
            r = build_model(construir_configuracion(familia, tratamiento),
                            X, y, nombre=etiqueta)
            oof = r["resumen_oof"]
            filas.append({
                "modelo": familia,
                "desbalance": tratamiento,
                # performance
                "auc_pr": r["auc_pr_media"],
                "ks": oof["ks"],
                "gini": oof["gini"],
                "recall_at_k": oof["recall_at_k"],
                # consistency
                "desv_folds": r["auc_pr_desv"],
                "peor_fold": min(r["auc_pr_folds"]),
                # scalability
                "seg_ajuste": r["seg_ajuste"],
                "seg_infer_1000": r["seg_prediccion_por_1000"],
                # calibracion, se arrastra para la etapa 7
                "brier": oof["brier"],
            })
            detalle[etiqueta] = r

    tabla = pd.DataFrame(filas).sort_values("auc_pr", ascending=False)
    return tabla.reset_index(drop=True), detalle


def seleccionar_mejor(tabla: pd.DataFrame, margen: float = 0.005) -> pd.Series:
    """Elige el modelo final aplicando los tres criterios en orden.

    No gana sin mas el AUC-PR mas alto. Las configuraciones que quedan dentro de
    `margen` de la mejor se consideran EMPATADAS, porque esa distancia es menor
    que la variabilidad entre folds y elegir por ella seria elegir por ruido.

    Entre las empatadas desempata la simplicidad, en dos ejes:

        familia       logistica sobre WoE produce un scorecard, con una tabla de
                      puntos que justifica la negacion ante el cliente y ante el
                      supervisor. En credito eso no es un empate tecnico, es una
                      ventaja
        tratamiento   ninguno < class_weight < smote. Cada escalon anade coste
                      sin aportar discriminacion medible: class_weight y smote
                      distorsionan las probabilidades y obligan a recalibrar
                      despues, y smote ademas duplica el tiempo de ajuste,
                      arrastra una dependencia mas y genera clientes sinteticos
                      interpolando entre valores WoE, que no corresponden a
                      ningun tramo real y por tanto no son explicables

    La desviacion entre folds solo entra como ultimo desempate. Antes ordenaba
    por ella y elegia smote por una diferencia de 0.0012, que es ruido, a cambio
    de un Brier cinco veces peor.
    """
    techo = tabla["auc_pr"].max()
    empatados = tabla[tabla["auc_pr"] >= techo - margen].copy()

    empatados["orden_familia"] = empatados["modelo"].map(
        {"logistica": 0, "boosting": 1, "bosque": 2})
    empatados["orden_tratamiento"] = empatados["desbalance"].map(
        {"ninguno": 0, "class_weight": 1, "smote": 2})

    empatados = empatados.sort_values(
        ["orden_familia", "orden_tratamiento", "desv_folds", "auc_pr"],
        ascending=[True, True, True, False])
    return empatados.iloc[0]


def figuras_comparativas(tabla: pd.DataFrame, detalle: dict, y,
                         destino: Path) -> list[Path]:
    """Graficos comparativos entre los modelos principales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    destino.mkdir(parents=True, exist_ok=True)
    guardados = []
    y = np.asarray(y).astype(int)
    tasa_base = y.mean()

    # --- 1. AUC-PR por configuracion, con su dispersion entre folds ---------
    fig, ax = plt.subplots(figsize=(9, 5))
    etiquetas = [f"{r.modelo}\n{r.desbalance}" for r in tabla.itertuples()]
    ax.bar(etiquetas, tabla["auc_pr"], yerr=tabla["desv_folds"],
           capsize=4, color="#4C72B0", edgecolor="#2A4A7B")
    ax.axhline(tasa_base, ls="--", c="#C44E52",
               label=f"azar (tasa base {tasa_base:.4f})")
    ax.set_ylabel("AUC-PR (media de 5 folds)")
    ax.set_title("Comparacion de modelos y tratamientos del desbalance\n"
                 "barras de error: desviacion entre folds (consistency)")
    ax.legend()
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    ruta = destino / "comparacion_auc_pr.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    guardados.append(ruta)

    # --- 2. Curvas precision-recall de los tres mejores ---------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for etiqueta in tabla.head(3).apply(
            lambda r: f"{r['modelo']}+{r['desbalance']}", axis=1):
        oof = detalle[etiqueta]["oof"]
        prec, rec, _ = precision_recall_curve(y, oof)
        ax.plot(rec, prec, lw=1.8,
                label=f"{etiqueta} (AUC-PR {detalle[etiqueta]['auc_pr_media']:.4f})")
    ax.axhline(tasa_base, ls="--", c="#C44E52", label="azar")
    ax.set_xlabel("Recall (mora capturada)")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas precision-recall, predicciones fuera de fold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ruta = destino / "curvas_precision_recall.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    guardados.append(ruta)

    return guardados


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

    # ------------------------------------------------------------------------
    # Etapas 4 y 5
    # ------------------------------------------------------------------------
    log.info("=" * 72)
    log.info("ETAPAS 4 Y 5 | Arboles, boosting y tratamiento del desbalance")
    log.info("=" * 72)

    comparaciones = {}
    for particion in ["estratificado", "temporal"]:
        log.info("-" * 72)
        log.info("[%s] rejilla de 3 familias x 3 tratamientos", particion)
        crudo = cargar(particion, con_features=False)
        y = 1 - crudo[TARGET]

        tabla, detalle = comparar_modelos(crudo, y)

        log.info("Tabla resumen, ordenada por AUC-PR:")
        for linea in tabla.to_string(index=False).split("\n"):
            log.info("   %s", linea)

        mejor = seleccionar_mejor(tabla)
        log.info("Seleccionado: %s + %s (AUC-PR %.4f, desviacion %.4f)",
                 mejor["modelo"], mejor["desbalance"], mejor["auc_pr"],
                 mejor["desv_folds"])

        figuras = figuras_comparativas(tabla, detalle, y,
                                       RUTA_MODELOS / "figuras" / particion)
        for f in figuras:
            log.info("Guardado: %s", f.relative_to(RUTA_RAIZ))

        comparaciones[particion] = {
            "tabla": tabla.to_dict(orient="records"),
            "seleccionado": mejor.to_dict(),
        }

        # El objeto del modelo elegido, que es lo que pide el enunciado. Se
        # guarda solo el de la particion estratificada: es la de referencia,
        # donde se selecciona. La temporal es prueba de estres y se usa en
        # model_evaluation.py.
        if particion == "estratificado":
            etiqueta = f"{mejor['modelo']}+{mejor['desbalance']}"
            ruta_modelo = RUTA_MODELOS / "modelo_seleccionado.joblib"
            joblib.dump(detalle[etiqueta]["modelo"], ruta_modelo)
            log.info("Guardado: %s  (%s)",
                     ruta_modelo.relative_to(RUTA_RAIZ), etiqueta)

    destino = RUTA_MODELOS / "etapas4y5_comparacion.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({
            "etapa": "4 y 5 - Familias de modelos y tratamiento del desbalance",
            "nota_test": "El conjunto de prueba NO se ha utilizado.",
            "nota_umbral": ("El ajuste de umbral no aparece en la tabla: AUC-PR "
                            "es independiente del umbral. Es la eleccion del "
                            "punto de operacion sobre el modelo ya entrenado, y "
                            "se decide en model_evaluation.py."),
            "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "ingenieria_dentro_del_cv": True,
            "resultados": comparaciones,
        }, f, indent=2, ensure_ascii=False, default=float)
    log.info("-" * 72)
    log.info("Guardado: %s", destino.relative_to(RUTA_RAIZ))
    log.info("=" * 72)

    return {"etapa3": resultados, "etapas4y5": comparaciones}


if __name__ == "__main__":
    main()
