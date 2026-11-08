"""
Pruebas del analisis de estabilidad
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

El bootstrap decide si la ventaja de AutoML es firme o puede ser azar. Estas
pruebas cuidan que sea pareado, que conserve la tasa de mora, que sea
reproducible y que la lectura del intervalo sea la correcta.

Los tests usan pocos remuestreos para correr en segundos; el script usa 1.000.
================================================================================
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import vertex_datos as vd
import vertex_estabilidad as es
import vertex_evaluacion as ve


def _datos(n: int = 400, tasa: float = 0.1, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    y = (rng.random(n) < tasa).astype(int)
    ruido = rng.random(n)
    return y, 0.3 * y + ruido, 0.6 * y + ruido


def test_el_remuestreo_conserva_cuantas_moras_hay():
    y = np.array([1] * 10 + [0] * 90)
    idx = es.indices_estratificados(y, np.random.default_rng(1))
    assert len(idx) == 100
    assert y[idx].sum() == 10


def test_el_bootstrap_es_reproducible_con_la_semilla():
    y, a, b = _datos()
    uno = es.bootstrap_pareado(y, a, b, remuestreos=30, semilla=7)
    otro = es.bootstrap_pareado(y, a, b, remuestreos=30, semilla=7)
    assert np.array_equal(uno["auc_pr_diferencia"], otro["auc_pr_diferencia"])


def test_es_pareado_un_modelo_contra_si_mismo_no_tiene_diferencia():
    y, a, _ = _datos()
    valores = es.bootstrap_pareado(y, a, a, remuestreos=30)
    assert np.all(valores["auc_pr_diferencia"] == 0)
    assert np.all(valores["roc_auc_diferencia"] == 0)


def test_un_modelo_claramente_mejor_tiene_ventaja_firme():
    y, a, b = _datos()
    resumen = es.resumir(es.bootstrap_pareado(y, a, b, remuestreos=60))
    assert resumen["auc_pr"]["ventaja_firme"] is True
    assert resumen["auc_pr"]["pct_gana_automl"] == 100.0
    bajo, alto = resumen["auc_pr"]["diferencia"]["ic95"]
    assert 0 < bajo <= resumen["auc_pr"]["diferencia"]["media"] <= alto


def test_si_el_intervalo_incluye_el_cero_la_ventaja_no_es_firme():
    valores = {f"{m}_{k}": np.linspace(-0.01, 0.03, 100)
               for m in es.METRICAS for k in ("python", "automl", "diferencia")}
    resumen = es.resumir(valores)
    assert resumen["auc_pr"]["ventaja_firme"] is False
    assert resumen["auc_pr"]["pct_gana_automl"] == 75.0
    assert resumen["auc_pr"]["pct_supera_el_ruido"] == pytest.approx(35.0, abs=1)


def test_main_mide_el_test_versionado_y_escribe_sus_salidas(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(es, "REMUESTREOS", 40)
    original = es.bootstrap_pareado
    monkeypatch.setattr(es, "bootstrap_pareado",
                        lambda y, a, b: original(y, a, b, remuestreos=40))
    assert es.main(["--salida", str(tmp_path)]) == 0
    resultado = json.loads((tmp_path / es.NOMBRE_JSON).read_text(encoding="utf-8"))
    assert resultado["metodo"]["pareado"] is True
    assert set(resultado["bootstrap"]) == {"auc_pr", "roc_auc"}
    assert (tmp_path / es.NOMBRE_FIGURA).exists()
    assert "gana AutoML en" in capsys.readouterr().out


def test_el_resultado_versionado_dice_que_la_ventaja_en_auc_pr_es_firme():
    resultado = json.loads((ve.RUTA_MODELOS / es.NOMBRE_JSON).read_text(encoding="utf-8"))
    auc_pr, roc_auc = resultado["bootstrap"]["auc_pr"], resultado["bootstrap"]["roc_auc"]
    assert resultado["metodo"]["remuestreos"] == 1000
    assert auc_pr["ventaja_firme"] is True and auc_pr["diferencia"]["ic95"][0] > 0
    assert roc_auc["ventaja_firme"] is False
    predicciones = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV)
    assert len(predicciones) == vd.TEST_ESPERADO
