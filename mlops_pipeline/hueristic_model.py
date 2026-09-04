"""
Modelo heuristico: el piso de referencia
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Contrato : reglas_negocio.py (bandas y criterio derivados del EDA)
Entrada  : data/processed/*_train.csv
Salida   : data/models/baseline_heuristico.json

POR QUE ESTE ARCHIVO VA ANTES QUE CUALQUIER MODELO

Antes de entrenar nada hay que responder una pregunta de negocio: que tan bien
se puede decidir SIN modelo. Si un algoritmo no supera una regla simple que un
analista podria aplicar a mano, no hay caso para desplegarlo. El coste de
mantener un modelo en produccion (monitoreo, reentrenamiento, validacion,
gobierno) solo se justifica si aporta sobre la alternativa barata.

DE QUE DEPENDE Y DE QUE NO

Depende de las PARTICIONES (data/processed/*_train.csv), porque el piso debe
medirse sobre exactamente los mismos datos que veran los modelos posteriores;
comparar sobre muestras distintas no significaria nada.

NO depende de la ingenieria de caracteristicas. No usa WoE, ni escalado, ni la
matriz de features: solo el puntaje crudo y el contrato. Que antes lo hiciera
fue un accidente de implementacion, no una necesidad.

QUE NO HACE ESTE ARCHIVO

No optimiza. El corte se deriva de un principio de negocio (PRINCIPIO_CORTE en
el contrato), no de maximizar una metrica: ajustar el umbral sobre train
convertiria la regla en un modelo entrenado y dejaria de ser un piso honesto.

El punto de corte se deriva EXCLUSIVAMENTE de train. El conjunto de prueba no
se toca hasta la evaluacion final, en model_evaluation.py.
================================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# El contrato del EDA vive en un modulo hermano. Se garantiza que el directorio
# de este archivo esta en sys.path para que la importacion funcione tanto al
# ejecutar el script directamente como al importarlo desde el notebook.
try:
    _DIR_MODULO = str(Path(__file__).resolve().parent)
except NameError:
    _DIR_MODULO = str(Path.cwd().resolve())
if _DIR_MODULO not in sys.path:
    sys.path.insert(0, _DIR_MODULO)

from reglas_negocio import (  # noqa: E402
    BANDAS_SCORE,
    MIN_POBLACION_BANDA_PCT,
    PRINCIPIO_CORTE,
    RECHAZAR_SIN_SCORE,
    VARIABLE_REGLA,
)


# CONFIGURACION


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("modelado")


def encontrar_raiz(marcador: str = "Base_de_datos.csv", max_niveles: int = 6) -> Path:
    """Sube por el arbol de directorios hasta encontrar el archivo marcador."""
    try:
        actual = Path(__file__).resolve().parent
    except NameError:
        actual = Path.cwd().resolve()
    for _ in range(max_niveles + 1):
        if (actual / marcador).exists():
            return actual
        if actual.parent == actual:
            break
        actual = actual.parent
    raise FileNotFoundError(f"No se encontro '{marcador}' subiendo desde {actual}.")


RUTA_RAIZ = encontrar_raiz()
RUTA_DATOS = RUTA_RAIZ / "data" / "processed"
RUTA_MODELOS = RUTA_RAIZ / "data" / "models"

CONFIG = json.load(open(RUTA_RAIZ / "config.json", encoding="utf-8"))
TARGET = CONFIG["target_variable"]
SEPARADOR = CONFIG["data"]["separator"]
ENCODING = CONFIG["data"]["encoding"]

# VARIABLE_REGLA, BANDAS_SCORE, MIN_POBLACION_BANDA_PCT, PRINCIPIO_CORTE y
# RECHAZAR_SIN_SCORE se importan del contrato del EDA. Este archivo APLICA la
# regla; no decide sus bandas ni su criterio.
#
# Version anterior de este bloque, corregida:
#   BANDAS_SCORE = [280, 600, 650, 700, 750, 800, 850, 950]
# El corte en 650 no proviene del EDA ni de la receta de la Fase 2, y generaba
# una banda de 8 registros. El comentario que lo acompaniaba afirmaba haber
# heredado las bandas de la Fase 2, cuando la Fase 2 hizo lo contrario:
# fusionarlas por inestables. Las metricas no cambian, porque la derivacion del
# corte ignora las bandas por debajo del 1% de la cartera, pero la cadena
# EDA -> reglas -> features queda restaurada.

# Resultados de la sonda de features de la Fase 2. Son el numero a superar.
AUC_PR_FASE2 = {"estratificado": 0.1424, "temporal": 0.1748}



# DERIVACION DEL PUNTO DE CORTE


def tabla_bandas(datos: pd.DataFrame) -> pd.DataFrame:
    """Tasa de mora observada por banda de score. Solo sobre train."""
    mora = 1 - datos[TARGET]
    bandas = pd.cut(datos[VARIABLE_REGLA], BANDAS_SCORE)
    tabla = pd.DataFrame({"banda": bandas, "mora": mora}).groupby(
        "banda", observed=True).agg(n=("mora", "size"), eventos=("mora", "sum"))
    tabla["tasa_%"] = (tabla["eventos"] / tabla["n"] * 100).round(2)
    tabla["%_cartera"] = (tabla["n"] / len(datos) * 100).round(1)
    return tabla


def derivar_corte(datos: pd.DataFrame,
                  min_poblacion: float = MIN_POBLACION_BANDA_PCT) -> tuple[int, str]:
    """Encuentra el corte donde la tasa de mora cruza la tasa base de la cartera.

    Es un principio de negocio, no una optimizacion

    Se ignoran las bandas con menos del 1% de la cartera: su tasa es inestable.
    """
    tabla = tabla_bandas(datos)
    tasa_base = (1 - datos[TARGET]).mean() * 100
    suficientes = tabla[tabla["%_cartera"] >= min_poblacion]

    # Recorriendo de mayor a menor score, el corte es el limite superior de la
    # primera banda cuya tasa ya supera la base.
    corte = None
    for banda in reversed(list(suficientes.index)):
        if suficientes.loc[banda, "tasa_%"] > tasa_base:
            corte = int(banda.right)
            break

    justificacion = (
        f"Todas las bandas por debajo de {corte} puntos presentan una tasa de mora "
        f"superior a la tasa base de la cartera ({tasa_base:.2f}%), y todas las bandas "
        f"por encima quedan por debajo de ella. El corte separa a los solicitantes "
        f"cuyo riesgo observado excede el promedio de la cartera."
    )
    return corte, justificacion



# EVALUACION


def aplicar_regla(datos: pd.DataFrame, corte: int) -> pd.Series:
    """Devuelve True donde la regla RECHAZA la solicitud.

    El tratamiento de los solicitantes sin score valido lo fija el contrato
    (RECHAZAR_SIN_SCORE): se rechazan porque su tasa de mora observada supera la
    base de la cartera, de modo que aprobarlos contradiria el mismo principio
    que define el corte.
    """
    score = datos[VARIABLE_REGLA]
    rechaza = score < corte
    if RECHAZAR_SIN_SCORE:
        rechaza = rechaza | score.isna()
    return rechaza


def evaluar_baseline(datos: pd.DataFrame, corte: int) -> dict:
    """Evalua dos baselines complementarios.

    A) LA REGLA como decision binaria se mide en su punto de
       operacion: a quien rechaza, cuanta mora captura y a que coste comercial.

    B) EL SCORE COMO ORDENADOR, usando score como puntuacion continua. Este si
       es comparable en AUC-PR contra los modelos de las etapas siguientes, y
       responde la pregunta de fondo: aporta un modelo multivariado algo sobre
       el score que la central de riesgo ya entrega?
    """
    mora = (1 - datos[TARGET]).astype(int)
    tasa_base = float(mora.mean())

    # --- A) La regla en su punto de operacion ---
    rechaza = aplicar_regla(datos, corte).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        mora, rechaza, average="binary", zero_division=0)
    aprobados = rechaza == 0

    regla = {
        "corte": int(corte),
        "pct_rechazado": round(float(rechaza.mean()) * 100, 2),
        "mora_capturada": int(((rechaza == 1) & (mora == 1)).sum()),
        "mora_total": int(mora.sum()),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "lift_precision": round(float(precision) / tasa_base, 2),
        "tasa_mora_aprobados_pct": round(float(mora[aprobados].mean()) * 100, 2),
        "tasa_mora_rechazados_pct": round(float(mora[~aprobados].mean()) * 100, 2),
    }

    # --- B) El score como ordenador ---
    score = datos[VARIABLE_REGLA]
    ordenador = (-score).fillna(-score.min())     # sin score -> tratado como el peor
    auc_pr = float(average_precision_score(mora, ordenador))
    ranker = {"auc_pr": round(auc_pr, 4),
              "lift_vs_azar": round(auc_pr / tasa_base, 2)}

    return {"tasa_base": round(tasa_base, 4), "regla": regla,
            "score_como_ranker": ranker}



# ORQUESTACION


def main() -> dict:
    log.info("=" * 72)
    log.info("FASE 3 - ETAPA 1 | Baseline heuristico")
    log.info("=" * 72)

    resultados = {}
    for particion in ["estratificado", "temporal"]:
        train = pd.read_csv(RUTA_DATOS / f"{particion}_train.csv",
                            sep=SEPARADOR, encoding=ENCODING)
        log.info("-" * 72)
        log.info("[%s] train = %s registros | tasa de mora = %.2f%%",
                 particion, len(train), (1 - train[TARGET]).mean() * 100)

        log.info("Tasa de mora por banda de score (derivada SOLO de train):")
        for linea in tabla_bandas(train).to_string().split("\n"):
            log.info("   %s", linea)

        corte, justificacion = derivar_corte(train)
        log.info("Punto de corte derivado: %s", corte)
        log.info("   %s", justificacion)

        ev = evaluar_baseline(train, corte)
        r, k = ev["regla"], ev["score_como_ranker"]

        log.info("A) Regla 'rechazar si score < %s o sin score':", corte)
        log.info("   rechaza el %.1f%% de las solicitudes", r["pct_rechazado"])
        log.info("   captura %s de %s casos de mora (recall %.1f%%)",
                 r["mora_capturada"], r["mora_total"], r["recall"] * 100)
        log.info("   precision %.2f%% (%.2fx la tasa base)",
                 r["precision"] * 100, r["lift_precision"])
        log.info("   mora entre aprobados %.2f%% vs rechazados %.2f%%",
                 r["tasa_mora_aprobados_pct"], r["tasa_mora_rechazados_pct"])
        log.info("B) Score como ordenador: AUC-PR = %.4f (lift %.2fx sobre el azar)",
                 k["auc_pr"], k["lift_vs_azar"])
        log.info("   Referencia Fase 2 (19 features): AUC-PR = %.4f "
                 "-> %.2fx el score solo",
                 AUC_PR_FASE2[particion], AUC_PR_FASE2[particion] / k["auc_pr"])

        ev["justificacion_corte"] = justificacion
        ev["auc_pr_fase2_referencia"] = AUC_PR_FASE2[particion]
        ev["mejora_features_sobre_score"] = round(
            AUC_PR_FASE2[particion] / k["auc_pr"], 2)
        resultados[particion] = ev

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    destino = RUTA_MODELOS / "baseline_heuristico.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({
            "etapa": "Fase 3 - Etapa 1: baseline heuristico",
            "variable_regla": VARIABLE_REGLA,
            "bandas_evaluadas": BANDAS_SCORE,
            "criterio_corte": PRINCIPIO_CORTE,
            "origen_bandas": "reglas_negocio.py (contrato derivado del EDA, celda 63)",
            "nota_test": "El conjunto de prueba NO se ha utilizado en esta etapa.",
            "resultados": resultados,
        }, f, indent=2, ensure_ascii=False)

    log.info("-" * 72)
    log.info("Guardado: data/models/baseline_heuristico.json")
    log.info("=" * 72)
    log.info("ETAPA 1 COMPLETADA. Piso de referencia fijado.")
    log.info("=" * 72)
    return resultados


if __name__ == "__main__":
    main()
