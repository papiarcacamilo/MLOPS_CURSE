"""
Las funciones que toman las decisiones del modelo
================================================================================
Cubre las funciones puras de model_training.py y model_evaluation.py.

Son pocas lineas, pero de ellas salio todo lo que se entrega: la metrica con la
que se comparo cada configuracion, la regla que eligio la logistica sobre el
bosque, el umbral de 0.0661 y el veredicto de que no hacia falta recalibrar. Si
alguien las modifica y rompe la metodologia, el resto de la suite no lo notaria:
los artefactos ya estan congelados y seguirian dando las mismas cifras.

No se reentrena nada. Cada test construye datos sinteticos con una respuesta
conocida de antemano, de modo que corren en milisegundos.

Senialado por danielCH26 en la revision de la PR6.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import model_evaluation as ev
import model_training as mt


# ==============================================================================
# summarize_classification: LA METRICA CON LA QUE SE COMPARO TODO
# ==============================================================================


def test_un_clasificador_perfecto_da_metricas_perfectas():
    y = np.array([0] * 90 + [1] * 10)
    p = np.where(y == 1, 0.95, 0.05)
    m = mt.summarize_classification(y, p)
    assert m["auc_pr"] == 1.0
    assert m["ks"] == 1.0
    assert m["gini"] == 1.0


def test_un_clasificador_al_azar_rinde_como_la_tasa_base():
    """El AUC-PR del azar es la tasa base, no 0.5: por eso se reporta el lift."""
    rng = np.random.default_rng(42)
    y = (rng.random(40_000) < 0.05).astype(int)
    p = rng.random(40_000)
    m = mt.summarize_classification(y, p)
    assert m["auc_pr"] == pytest.approx(m["tasa_base"], abs=0.01)
    assert m["lift_vs_azar"] == pytest.approx(1.0, abs=0.15)
    assert m["gini"] == pytest.approx(0.0, abs=0.05)


def test_invertir_las_probabilidades_invierte_el_gini():
    rng = np.random.default_rng(7)
    y = (rng.random(5_000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.15, 5_000), 0, 1)
    directo = mt.summarize_classification(y, p)["gini"]
    invertido = mt.summarize_classification(y, 1 - p)["gini"]
    assert directo > 0.5
    assert invertido == pytest.approx(-directo, abs=1e-3)


def test_etiquetas_y_probabilidades_de_distinto_tamanio_fallan():
    with pytest.raises(ValueError, match="no coinciden"):
        mt.summarize_classification([0, 1, 0], [0.1, 0.9])


def test_el_recall_at_k_cuenta_la_mora_dentro_del_k_peor():
    """Es la metrica que traduce el modelo a la decision de negocio.

    10 solicitudes ordenadas de mayor a menor riesgo, con mora en la primera y
    en la sexta. Rechazar el 20% peor (2 solicitudes) atrapa solo la primera.
    """
    p = np.linspace(0.9, 0.0, 10)
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    m = mt.summarize_classification(y, p, k=0.2)
    assert m["recall_at_k"] == 0.5
    assert m["precision_at_k"] == 0.5
    assert m["lift_at_k"] == 2.5


def test_precision_y_recall_se_miden_en_el_umbral_dado():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.8, 0.3, 0.6, 0.2, 0.1])
    m = mt.summarize_classification(y, p, umbral=0.5)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["pct_marcado"] == 40.0


# ==============================================================================
# seleccionar_mejor: LA REGLA QUE ELIGIO EL MODELO
# ==============================================================================


def _tabla(filas: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(filas, columns=["modelo", "desbalance", "auc_pr", "desv_folds"])


def test_dentro_del_margen_gana_la_logistica_sin_tratar():
    """El caso real de la particion temporal.

    El bosque saca 0.0038 mas, una distancia menor que la variacion entre folds.
    Elegirlo seria elegir por ruido y perder el scorecard.
    """
    tabla = _tabla([("bosque", "ninguno", 0.1703, 0.0384),
                    ("logistica", "ninguno", 0.1665, 0.0300),
                    ("logistica", "class_weight", 0.1646, 0.0272)])
    elegido = mt.seleccionar_mejor(tabla)
    assert (elegido["modelo"], elegido["desbalance"]) == ("logistica", "ninguno")


def test_fuera_del_margen_gana_el_mejor_auc_pr():
    """La simplicidad desempata, pero no compensa una diferencia real."""
    tabla = _tabla([("bosque", "ninguno", 0.2000, 0.0300),
                    ("logistica", "ninguno", 0.1500, 0.0100)])
    assert mt.seleccionar_mejor(tabla)["modelo"] == "bosque"


def test_entre_tratamientos_empatados_gana_no_tratar_el_desbalance():
    """SMOTE con mas AUC-PR y menos variacion pierde si esta dentro del margen."""
    tabla = _tabla([("logistica", "smote", 0.1400, 0.0100),
                    ("logistica", "ninguno", 0.1380, 0.0200)])
    assert mt.seleccionar_mejor(tabla)["desbalance"] == "ninguno"


# ==============================================================================
# CALIBRACION Y UMBRAL
# ==============================================================================


def _calibradas(n: int = 50_000, semilla: int = 3):
    rng = np.random.default_rng(semilla)
    p = rng.uniform(0.0, 0.2, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_probabilidades_calibradas_dan_error_de_calibracion_bajo():
    y, p = _calibradas()
    c = ev.evaluar_calibracion(y, p)
    assert c["ece"] < 0.01
    assert abs(c["sesgo_global"]) < 0.005


def test_probabilidades_infladas_se_detectan():
    """Es el caso de class_weight: ordena igual, pero promete otra mora."""
    y, p = _calibradas()
    c = ev.evaluar_calibracion(y, np.clip(p * 2, 0, 1))
    assert c["ece"] > 0.05
    assert c["sesgo_global"] > 0.05


def test_la_curva_de_calibracion_agrupa_por_cuantiles():
    """Por cuantiles y no por intervalos fijos: todos los grupos pesan igual."""
    y, p = _calibradas(n=10_000)
    curva = ev.curva_calibracion(y, p, n_grupos=10)
    assert len(curva) == 10
    assert curva["n"].min() == curva["n"].max() == 1_000


def test_el_umbral_rechaza_el_volumen_de_la_regla():
    """El punto de operacion se fija por volumen: el 20.9% de la regla heuristica."""
    rng = np.random.default_rng(11)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p * 0.1).astype(int)
    u = ev.elegir_umbral(y, p)
    assert u["pct_rechazado"] == pytest.approx(mt.K_OPERATIVO * 100, abs=0.1)
