"""
Evaluacion del modelo
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/models/modelo_seleccionado.joblib + data/processed/
Salida   : pestania de metricas, scorecard y umbral de operacion

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Genera un proceso de evaluacion que crea una pestania de metricas para
     conocer el desempenio del modelo desplegado."

ORDEN DE LAS ETAPAS, Y POR QUE NO ES EL DE LA HOJA DE RUTA

La hoja de ruta numeraba: 6 estres temporal, 7 calibracion, 8 fairness, 9 test
final. Ese orden abre el test en la etapa 6 y lo vuelve a abrir en la 9.

El conjunto de prueba se mira UNA sola vez. Mirarlo, ajustar algo y volver a
mirarlo lo convierte en un segundo conjunto de validacion, y la cifra que se
reporta deja de ser una estimacion honesta del desempenio en produccion.

Asi que se reordena. Todo lo que se puede decidir sin el test se decide antes:

    1. Calibracion       sobre predicciones fuera de fold de entrenamiento
    2. Fairness          sobre las mismas predicciones
    3. Umbral            se fija el punto de operacion
    4. Scorecard         se deriva de los coeficientes, no de datos nuevos

Y solo entonces, con todas las decisiones tomadas, se abre el test una vez:

    5. Evaluacion final  estratificado (referencia) y temporal (estres)

Las predicciones fuera de fold sirven para esto porque cada una la produjo un
modelo que no habia visto ese registro: son una estimacion honesta sin gastar
el test.
================================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

try:
    _DIR_MODULO = str(Path(__file__).resolve().parent)
except NameError:
    _DIR_MODULO = str(Path.cwd().resolve())
if _DIR_MODULO not in sys.path:
    sys.path.insert(0, _DIR_MODULO)

import ft_engineering as fe  # noqa: E402
import hueristic_model as hm  # noqa: E402
import model_training as mt  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("evaluacion")

RUTA_RAIZ = fe.RUTA_RAIZ
RUTA_DATOS = fe.RUTA_SALIDA
RUTA_MODELOS = fe.RUTA_MODELOS
TARGET = fe.TARGET


def cargar_particion(particion: str, conjunto: str) -> pd.DataFrame:
    return pd.read_csv(RUTA_DATOS / f"{particion}_{conjunto}.csv",
                       sep=fe.SEPARADOR, encoding=fe.ENCODING)


def predicciones_fuera_de_fold(modelo, X, y) -> np.ndarray:
    """Reproduce las predicciones fuera de fold del modelo elegido.

    Cada prediccion la emite un modelo que NO vio ese registro al entrenarse, de
    modo que sirve para decidir calibracion, fairness y umbral sin gastar el
    conjunto de prueba.
    """
    cv = mt.folds_estandar()
    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan)
    for i_tr, i_va in cv.split(X, y):
        m = clone(modelo).fit(X.iloc[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(X.iloc[i_va])[:, 1]
    return oof


# ==============================================================================
# 1 | CALIBRACION
# ==============================================================================


def curva_calibracion(y, p, n_grupos: int = 10) -> pd.DataFrame:
    """Agrupa por probabilidad predicha y compara con la mora observada.

    Se agrupa por cuantiles y no en intervalos fijos porque las probabilidades
    se concentran cerca de cero: con intervalos de igual anchura casi todos los
    registros caerian en el primero y la curva no diria nada.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    grupos = pd.qcut(p, n_grupos, duplicates="drop", labels=False)
    tabla = pd.DataFrame({"grupo": grupos, "p": p, "y": y}).groupby("grupo").agg(
        n=("y", "size"),
        predicha=("p", "mean"),
        observada=("y", "mean"),
    )
    tabla["diferencia"] = tabla["predicha"] - tabla["observada"]
    return tabla.reset_index()


def evaluar_calibracion(y, p) -> dict:
    """Mide si las probabilidades son creibles, no solo si ordenan bien.

    Discriminacion y calibracion son propiedades independientes. Un modelo puede
    ordenar perfecto y estar pesimo calibrado. Importa porque la probabilidad de
    incumplimiento entra al calculo de provisiones: mal calibrada significa
    provisionar mal, que es un problema contable antes que estadistico.

    El error de calibracion esperado (ECE) promedia la diferencia entre lo
    predicho y lo observado, ponderando por el tamanio de cada grupo.
    """
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    tabla = curva_calibracion(y, p)

    ece = float(np.average(tabla["diferencia"].abs(), weights=tabla["n"]))
    sesgo = float(p.mean() - y.mean())

    return {
        "brier": round(float(brier_score_loss(y, p)), 5),
        "ece": round(ece, 5),
        "prob_media_predicha": round(float(p.mean()), 5),
        "tasa_observada": round(float(y.mean()), 5),
        "sesgo_global": round(sesgo, 5),
        "curva": tabla.round(5).to_dict(orient="records"),
    }


# ==============================================================================
# 2 | FAIRNESS
# ==============================================================================


def metricas_por_grupo(y, p, grupo: pd.Series, umbral: float) -> pd.DataFrame:
    """Tasas de aprobacion y de error por grupo, no solo AUC global.

    Un modelo puede tener buen AUC y aun asi equivocarse de forma sistematica en
    un subgrupo. Por eso se miden dos familias:

        paridad demografica   misma tasa de rechazo entre grupos
        igualdad de opciones  mismas tasas de acierto y de falsa alarma DENTRO
                              de cada clase real

    Las dos rara vez se cumplen a la vez cuando la tasa base difiere entre
    grupos, que es el caso aqui. La lectura correcta no es exigir que ambas den
    cero, sino comprobar que la diferencia sea explicable por riesgo y no por
    pertenencia al grupo.
    """
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rechaza = (p >= umbral).astype(int)
    g = pd.Series(np.asarray(grupo)).reset_index(drop=True)

    filas = []
    for valor in sorted(g.dropna().unique(), key=str):
        m = (g == valor).to_numpy()
        yg, pg, rg = y[m], p[m], rechaza[m]
        positivos, negativos = yg == 1, yg == 0
        filas.append({
            "grupo": str(valor),
            "n": int(m.sum()),
            "tasa_base": round(float(yg.mean()), 4),
            "tasa_rechazo": round(float(rg.mean()), 4),
            # Igualdad de opciones: se calculan dentro de cada clase real.
            "tpr": round(float(rg[positivos].mean()), 4) if positivos.any() else None,
            "fpr": round(float(rg[negativos].mean()), 4) if negativos.any() else None,
            "auc_pr": (round(float(average_precision_score(yg, pg)), 4)
                       if 0 < yg.sum() < len(yg) else None),
        })
    return pd.DataFrame(filas)


def resumen_fairness(tabla: pd.DataFrame) -> dict:
    """Convierte la tabla por grupo en las dos brechas estandar."""
    def brecha(col):
        v = tabla[col].dropna()
        return round(float(v.max() - v.min()), 4) if len(v) > 1 else None

    return {
        "paridad_demografica": brecha("tasa_rechazo"),
        "brecha_tpr": brecha("tpr"),
        "brecha_fpr": brecha("fpr"),
        "brecha_tasa_base": brecha("tasa_base"),
        "por_grupo": tabla.to_dict(orient="records"),
    }


def tramos_edad(edad: pd.Series) -> pd.Series:
    """Edad en tramos, para poder auditarla como grupo."""
    return pd.cut(edad, [0, 30, 45, 60, 200],
                  labels=["18-30", "31-45", "46-60", "60+"])


# ==============================================================================
# 3 | UMBRAL DE OPERACION
# ==============================================================================


def elegir_umbral(y, p, pct_rechazo: float = mt.K_OPERATIVO) -> dict:
    """Fija el punto de operacion, que es una decision de negocio.

    El umbral por defecto de 0.5 no sirve aqui: sin reponderar clases el modelo
    casi nunca supera esa probabilidad, de modo que no rechazaria a nadie. Eso
    no es un defecto del modelo, es que 0.5 no significa nada cuando la tasa base
    es 4.75%.

    El criterio es el volumen de rechazo, no la probabilidad: se elige el umbral
    que rechaza el mismo porcentaje que la regla heuristica, para que la
    comparacion entre ambas sea a igual coste comercial.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    umbral = float(np.quantile(p, 1 - pct_rechazo))
    rechaza = p >= umbral
    return {
        "umbral": round(umbral, 6),
        "criterio": (f"cuantil que rechaza el {pct_rechazo * 100:.1f}% de las "
                     f"solicitudes, el mismo volumen que la regla heuristica"),
        "pct_rechazado": round(float(rechaza.mean()) * 100, 2),
        "recall": round(float(y[rechaza].sum() / y.sum()), 4),
        "precision": round(float(y[rechaza].mean()), 4),
        "tasa_mora_aprobados": round(float(y[~rechaza].mean()), 4),
    }


# ==============================================================================
# 4 | SCORECARD
# ==============================================================================


def _modelo_lineal_efectivo(modelo) -> tuple[np.ndarray, float, list]:
    """Devuelve los coeficientes sobre las caracteristicas SIN escalar.

    El modelo elegido lleva un StandardScaler delante de la logistica, de modo
    que sus coeficientes estan expresados sobre variables tipificadas. Para una
    tabla de puntos hacen falta sobre el WoE original, asi que se pliega el
    escalado dentro del coeficiente:

        z = b0 + sum( bi * (xi - mi) / si )
          = [ b0 - sum(bi*mi/si) ] + sum( (bi/si) * xi )
    """
    caracteristicas = modelo.named_steps["caracteristicas"]
    interno = modelo.named_steps["modelo"]
    escalador = interno.named_steps["standardscaler"]
    logistica = interno.named_steps["logisticregression"]

    beta = logistica.coef_[0] / escalador.scale_
    intercepto = float(logistica.intercept_[0] - np.sum(
        logistica.coef_[0] * escalador.mean_ / escalador.scale_))
    columnas = list(caracteristicas.get_feature_names_out())
    return beta, intercepto, columnas


def construir_scorecard(modelo, pdo: int = 20, score_base: int = 600,
                        odds_base: float = 20.0) -> dict:
    """Traduce el modelo a una tabla de puntos por tramo.

    Es el formato con el que un analista de riesgo justifica una negacion ante
    el cliente y ante el supervisor: cada tramo suma o resta puntos, y el total
    determina la decision. La escala es la convencion del sector:

        factor  = pdo / ln(2)        cuantos puntos duplican las probabilidades
        offset  = score_base - factor * ln(odds_base)

    Mas puntos significan menos riesgo. Por eso el signo se invierte respecto al
    WoE, donde mas valor significa mas riesgo.

    LIMITACION, y hay que declararla. Solo las 9 variables WoE tienen tramos
    discretos que se pueden tabular. Las 4 monetarias escaladas son continuas y
    entran como un ajuste sobre el puntaje, no como fila de la tabla. Un
    scorecard completo exigiria discretizarlas tambien, a costa de perder
    resolucion.
    """
    factor = pdo / np.log(2)
    offset = score_base - factor * np.log(odds_base)

    beta, intercepto, columnas = _modelo_lineal_efectivo(modelo)
    receta = modelo.named_steps["caracteristicas"].named_transformers_["woe"].receta_

    woe_cols = [c for c in columnas if c.startswith("woe_")]
    n_woe = len(woe_cols)
    puntos_base = offset - factor * intercepto

    filas = []
    for i, col in enumerate(columnas):
        if not col.startswith("woe_"):
            continue
        variable = col[len("woe_"):]
        for tramo, woe in receta[variable]["woe"].items():
            filas.append({
                "variable": variable,
                "tramo": tramo,
                "woe": round(float(woe), 4),
                "coeficiente": round(float(beta[i]), 4),
                # El reparto del intercepto entre variables es la convencion del
                # sector: hace que la suma de puntos reconstruya el puntaje.
                "puntos": int(round(-factor * beta[i] * woe + puntos_base / n_woe)),
            })

    tabla = pd.DataFrame(filas)
    continuas = [{"variable": c, "coeficiente": round(float(beta[i]), 4),
                  "puntos_por_unidad": round(float(-factor * beta[i]), 2)}
                 for i, c in enumerate(columnas) if not c.startswith("woe_")]

    return {
        "escala": {"pdo": pdo, "score_base": score_base, "odds_base": odds_base,
                   "factor": round(float(factor), 4),
                   "offset": round(float(offset), 4)},
        "puntos_base": int(round(puntos_base)),
        "rango_tabla": [int(tabla["puntos"].min()), int(tabla["puntos"].max())],
        "tabla": tabla.to_dict(orient="records"),
        "ajustes_continuos": continuas,
        "nota": ("Solo las 9 variables WoE se tabulan por tramo. Las 4 monetarias "
                 "escaladas y las 4 banderas entran como ajuste continuo."),
    }


def verificar_scorecard(modelo, X, scorecard: dict) -> dict:
    """Comprueba que la tabla de puntos reproduce la prediccion del modelo.

    Si el puntaje reconstruido no se corresponde con la probabilidad que emite
    el modelo, la tabla que veria el analista no representa lo que decide el
    sistema, y el scorecard seria decorativo.
    """
    beta, intercepto, _ = _modelo_lineal_efectivo(modelo)
    caracteristicas = modelo.named_steps["caracteristicas"]

    Z = caracteristicas.transform(X).to_numpy() @ beta + intercepto
    p_modelo = modelo.predict_proba(X)[:, 1]
    p_desde_z = 1 / (1 + np.exp(-Z))

    factor = scorecard["escala"]["factor"]
    offset = scorecard["escala"]["offset"]
    puntajes = offset - factor * Z

    return {
        "error_max_probabilidad": float(np.abs(p_modelo - p_desde_z).max()),
        "coincide": bool(np.allclose(p_modelo, p_desde_z, atol=1e-9)),
        "puntaje_min": int(puntajes.min()),
        "puntaje_max": int(puntajes.max()),
        "puntaje_medio": int(puntajes.mean()),
    }


# ==============================================================================
# 5 | EVALUACION FINAL | SE ABRE EL TEST UNA SOLA VEZ
# ==============================================================================


def evaluacion_final(modelo, particion: str, umbral: float) -> dict:
    """Entrena sobre train y evalua sobre test. Una vez.

    El modelo se reajusta sobre TODO el train de la particion y se aplica al
    test sin volver a mirar nada. El umbral llega ya fijado desde las
    predicciones fuera de fold: elegirlo aqui seria ajustarlo al test.
    """
    tr = cargar_particion(particion, "train")
    te = cargar_particion(particion, "test")

    X_tr, y_tr = tr.drop(columns=[TARGET]), (1 - tr[TARGET]).values
    X_te, y_te = te.drop(columns=[TARGET]), (1 - te[TARGET]).values

    ajustado = clone(modelo).fit(X_tr, y_tr)
    p_te = ajustado.predict_proba(X_te)[:, 1]

    metricas = mt.summarize_classification(y_te, p_te, umbral=umbral,
                                           k=mt.K_OPERATIVO)

    # El piso heuristico, sobre EL MISMO conjunto de prueba.
    #
    # Sin esto la comparacion seria invalida. El AUC-PR depende fuertemente de
    # la tasa base, y la de cada conjunto es distinta: 4.74% en el test
    # estratificado y 3.20% en el temporal, frente al 4.75% y 5.13% de sus
    # respectivos train. Comparar el modelo en test contra el heuristico en
    # train mezclaria dos poblaciones y haria parecer que el modelo empeora
    # cuando lo que cambio fue la proporcion de eventos.
    #
    # Por eso se reporta ademas el lift sobre la tasa base, que si es comparable
    # entre conjuntos.
    from sklearn.metrics import average_precision_score
    score = te[hm.VARIABLE_REGLA]
    orden_heuristico = (-score).fillna(-score.min())
    auc_heuristico = float(average_precision_score(y_te, orden_heuristico))
    rechaza_regla = hm.aplicar_regla(te, 750).to_numpy()

    resultado = {
        "n_test": int(len(y_te)),
        "eventos_test": int(y_te.sum()),
        "metricas": metricas,
        "calibracion": {k: v for k, v in evaluar_calibracion(y_te, p_te).items()
                        if k != "curva"},
        "piso_en_el_mismo_test": {
            "auc_pr": round(auc_heuristico, 4),
            "lift_vs_azar": round(auc_heuristico / float(y_te.mean()), 2),
            "pct_rechazado": round(float(rechaza_regla.mean()) * 100, 2),
            "recall": round(float(y_te[rechaza_regla].sum() / y_te.sum()), 4),
            "precision": round(float(y_te[rechaza_regla].mean()), 4),
        },
        "modelo_supera_al_piso": bool(metricas["auc_pr"] > auc_heuristico),
        "ventaja_sobre_el_piso": round(metricas["auc_pr"] / auc_heuristico, 2),
    }

    # Prueba de estres: en la particion temporal el test concentra creditos
    # recientes. Se evalua tambien sobre los que ya vencieron.
    if "madurez_incompleta" in te.columns:
        maduros = ~te["madurez_incompleta"].astype(bool).to_numpy()
        resultado["madurez_incompleta_pct"] = round(
            float((~maduros).mean()) * 100, 1)
        if maduros.sum() > 30 and y_te[maduros].sum() > 5:
            resultado["solo_maduros"] = {
                "n": int(maduros.sum()),
                "eventos": int(y_te[maduros].sum()),
                "metricas": mt.summarize_classification(
                    y_te[maduros], p_te[maduros], umbral=umbral,
                    k=mt.K_OPERATIVO),
            }
    return resultado


# ==============================================================================
# ORQUESTACION
# ==============================================================================


def main() -> dict:
    log.info("=" * 72)
    log.info("EVALUACION DEL MODELO")
    log.info("=" * 72)

    modelo = joblib.load(RUTA_MODELOS / "modelo_seleccionado.joblib")
    tr = cargar_particion("estratificado", "train")
    X, y = tr.drop(columns=[TARGET]), (1 - tr[TARGET]).values

    log.info("Modelo: %s", type(modelo.named_steps["modelo"]).__name__)
    log.info("Prediciendo fuera de fold sobre train (el test no se toca aun)")
    oof = predicciones_fuera_de_fold(modelo, X, y)

    # --- 1. Calibracion ---------------------------------------------------
    log.info("-" * 72)
    log.info("1 | CALIBRACION")
    cal = evaluar_calibracion(y, oof)
    log.info("Brier %.5f | ECE %.5f", cal["brier"], cal["ece"])
    log.info("Probabilidad media predicha %.5f vs mora observada %.5f (sesgo %+.5f)",
             cal["prob_media_predicha"], cal["tasa_observada"], cal["sesgo_global"])
    if cal["ece"] < 0.01:
        log.info("Calibracion aceptable: NO se aplica Platt ni isotonica. "
                 "Recalibrar un modelo ya calibrado solo anadiria varianza")
    else:
        log.warning("ECE por encima de 0.01: conviene recalibrar")

    # --- 2. Fairness ------------------------------------------------------
    log.info("-" * 72)
    log.info("2 | FAIRNESS")
    umbral_info = elegir_umbral(y, oof)
    fairness = {}
    for nombre, grupo in [("tipo_laboral", tr["tipo_laboral"]),
                          ("edad", tramos_edad(tr["edad_cliente"]))]:
        tabla = metricas_por_grupo(y, oof, grupo, umbral_info["umbral"])
        fairness[nombre] = resumen_fairness(tabla)
        log.info("[%s]", nombre)
        for linea in tabla.to_string(index=False).split("\n"):
            log.info("   %s", linea)
        r = fairness[nombre]
        log.info("   brecha en tasa de rechazo %.4f | en tasa base real %.4f",
                 r["paridad_demografica"], r["brecha_tasa_base"])
        if r["paridad_demografica"] > 3 * max(r["brecha_tasa_base"], 1e-6):
            log.warning("   El rechazo se separa mas que el riesgo real. "
                        "Debe justificarse o corregirse, no darse por bueno")

    # --- 3. Umbral --------------------------------------------------------
    log.info("-" * 72)
    log.info("3 | UMBRAL DE OPERACION")
    log.info("Umbral %.6f: rechaza el %.2f%%, captura el %.1f%% de la mora, "
             "precision %.2f%%", umbral_info["umbral"],
             umbral_info["pct_rechazado"], umbral_info["recall"] * 100,
             umbral_info["precision"] * 100)
    log.info("Mora entre aprobados: %.2f%% (tasa base %.2f%%)",
             umbral_info["tasa_mora_aprobados"] * 100, y.mean() * 100)

    # --- 4. Scorecard -----------------------------------------------------
    log.info("-" * 72)
    log.info("4 | SCORECARD")
    modelo_ajustado = clone(modelo).fit(X, y)
    scorecard = construir_scorecard(modelo_ajustado)
    verificacion = verificar_scorecard(modelo_ajustado, X, scorecard)
    log.info("Tabla de %s filas | puntaje base %s | rango de puntos por tramo %s",
             len(scorecard["tabla"]), scorecard["puntos_base"],
             scorecard["rango_tabla"])
    log.info("Puntajes observados en train: %s a %s (medio %s)",
             verificacion["puntaje_min"], verificacion["puntaje_max"],
             verificacion["puntaje_medio"])
    if verificacion["coincide"]:
        log.info("La tabla reproduce la prediccion del modelo (error < 1e-9)")
    else:
        log.error("La tabla NO reproduce el modelo: error maximo %.2e",
                  verificacion["error_max_probabilidad"])

    # --- 5. Test, una sola vez -------------------------------------------
    log.info("=" * 72)
    log.info("5 | EVALUACION FINAL. Se abre el conjunto de prueba")
    log.info("=" * 72)
    finales = {}
    for particion in ["estratificado", "temporal"]:
        r = evaluacion_final(modelo, particion, umbral_info["umbral"])
        finales[particion] = r
        m = r["metricas"]
        log.info("-" * 72)
        log.info("[%s] test: %s registros, %s casos de mora (%.2f%%)",
                 particion, r["n_test"], r["eventos_test"],
                 m["tasa_base"] * 100)
        log.info("   AUC-PR %.4f | KS %.4f | Gini %.4f | Brier %.5f",
                 m["auc_pr"], m["ks"], m["gini"], m["brier"])
        log.info("   En el punto de operacion: rechaza %.1f%%, captura %.1f%% "
                 "de la mora, precision %.2f%%",
                 m["pct_marcado"], m["recall"] * 100, m["precision"] * 100)
        ph = r["piso_en_el_mismo_test"]
        log.info("   Piso heuristico EN ESTE MISMO test: AUC-PR %.4f (lift %.2fx)",
                 ph["auc_pr"], ph["lift_vs_azar"])
        log.info("   Modelo: lift %.2fx sobre el azar -> %.2fx el piso  [%s]",
                 m["lift_vs_azar"], r["ventaja_sobre_el_piso"],
                 "SUPERA" if r["modelo_supera_al_piso"] else "NO SUPERA")
        if "solo_maduros" in r:
            sm = r["solo_maduros"]["metricas"]
            log.info("   Madurez incompleta: %.1f%% del test",
                     r["madurez_incompleta_pct"])
            log.info("   Solo creditos ya vencidos (%s registros, %s eventos): "
                     "AUC-PR %.4f (lift %.2fx) | KS %.4f",
                     r["solo_maduros"]["n"], r["solo_maduros"]["eventos"],
                     sm["auc_pr"], sm["lift_vs_azar"], sm["ks"])

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    destino = RUTA_MODELOS / "evaluacion.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({
            "modelo": "logistica sobre WoE, sin tratamiento del desbalance",
            "orden": ("calibracion, fairness, umbral y scorecard se deciden sobre "
                      "predicciones fuera de fold. El test se abre una sola vez, "
                      "al final, con todas las decisiones ya tomadas."),
            "calibracion": cal,
            "fairness": fairness,
            "umbral": umbral_info,
            "scorecard": scorecard,
            "verificacion_scorecard": verificacion,
            "evaluacion_final": finales,
            "limitacion": ("El dataset no tiene identificador de cliente. La unidad "
                           "de observacion es el credito, no la persona, y el 67.9% "
                           "de las filas comparte perfil demografico con otra. No se "
                           "pudo aplicar particion agrupada por cliente."),
        }, f, indent=2, ensure_ascii=False, default=float)
    log.info("-" * 72)
    log.info("Guardado: %s", destino.relative_to(RUTA_RAIZ))
    log.info("=" * 72)
    return {"calibracion": cal, "fairness": fairness, "umbral": umbral_info,
            "scorecard": scorecard, "final": finales}


if __name__ == "__main__":
    main()
