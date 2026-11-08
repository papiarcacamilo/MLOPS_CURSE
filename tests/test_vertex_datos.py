"""
Pruebas de la tabla de modelado para Vertex AI
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

La tabla es la base de la comparacion entre Python, AutoML y el pipeline propio.
Si el test no fuera el mismo o se colara una columna, la comparacion dejaria de
ser justa sin que nada fallara. Estas pruebas lo impiden.

Ninguna prueba sube datos: la subida se reemplaza por una funcion falsa.
================================================================================
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import vertex_datos as vd

ENTRADAS = ["capital_prestado", "plazo_meses"]


def _particion(n: int, moras: int) -> pd.DataFrame:
    return pd.DataFrame({
        "capital_prestado": range(n),
        "plazo_meses": [12] * n,
        "fecha_prestamo": ["2025-01-01"] * n,
        "saldo_mora": [0] * n,
        "Pago_atiempo": [0] * moras + [1] * (n - moras),
    })


def _tabla(n_train: int = 200, moras_train: int = 20,
           n_test: int = 50, moras_test: int = 5) -> pd.DataFrame:
    return vd.construir_tabla(_particion(n_train, moras_train),
                              _particion(n_test, moras_test), ENTRADAS,
                              objetivo_origen="Pago_atiempo", semilla=42,
                              proporcion_validacion=0.1)


def test_la_tabla_solo_lleva_las_entradas_el_objetivo_y_la_particion():
    tabla = _tabla()
    assert list(tabla.columns) == [vd.ID, *ENTRADAS, vd.OBJETIVO, vd.PARTICION]


def test_mora_es_la_clase_positiva():
    tabla = _tabla()
    assert tabla[vd.OBJETIVO].sum() == 25


def test_el_test_queda_entero_y_la_validacion_sale_del_train():
    tabla = _tabla()
    conteo = tabla[vd.PARTICION].value_counts()
    assert conteo["TEST"] == 50
    assert conteo["TRAIN"] + conteo["VALIDATE"] == 200
    assert conteo["VALIDATE"] == 20
    assert tabla[tabla[vd.PARTICION] == "VALIDATE"][vd.OBJETIVO].sum() == 2


def test_la_tabla_real_cumple_las_reglas_de_la_comparacion():
    config = vd.cargar_config()
    entradas = vd.entradas_del_modelo()
    tabla = vd.construir_tabla(
        vd.leer_particion("estratificado_train", config),
        vd.leer_particion("estratificado_test", config),
        entradas, objetivo_origen=config["target_variable"],
        semilla=config["random_state"],
        proporcion_validacion=config["vertex"]["proporcion_validacion"])
    assert vd.COLUMNA_FECHA not in entradas
    assert len(entradas) == 12
    assert vd.verificar_tabla(tabla, entradas) == []


def test_un_test_distinto_se_rechaza():
    problemas = vd.verificar_tabla(_tabla(), ENTRADAS)
    assert any("el test no es el de Python" in p for p in problemas)
    assert any("no suman el train" in p for p in problemas)


@pytest.mark.parametrize("columna, valor, mensaje", [
    (vd.PARTICION, "OTRA", "particion no validos"),
    (vd.OBJETIVO, 2, "no es binario"),
    (vd.ID, "tr00001", "repetidos"),
])
def test_cada_defecto_se_detecta(columna, valor, mensaje):
    tabla = _tabla()
    tabla.loc[0, columna] = valor
    assert any(mensaje in p for p in vd.verificar_tabla(tabla, ENTRADAS))


def test_una_columna_de_mas_se_rechaza():
    tabla = _tabla()
    tabla["saldo_mora"] = 0
    assert vd.verificar_tabla(tabla, ENTRADAS)[0].startswith("columnas inesperadas")


def test_validacion_sin_moras_se_rechaza():
    tabla = _tabla()
    tabla.loc[tabla[vd.PARTICION] == "VALIDATE", vd.OBJETIVO] = 0
    assert "la validacion no tiene moras" in vd.verificar_tabla(tabla, ENTRADAS)


def test_main_sin_subir_no_toca_bigquery(monkeypatch, capsys):
    llamadas = []
    monkeypatch.setattr(vd, "subir", lambda *a: llamadas.append(a))
    assert vd.main([]) == 0
    assert llamadas == []
    assert "APROBADO" in capsys.readouterr().out


def test_main_con_subir_reemplaza_la_tabla_configurada(monkeypatch, capsys):
    llamadas = []
    falso = types.SimpleNamespace(to_gbq=lambda *a, **k: llamadas.append((a, k)))
    monkeypatch.setitem(sys.modules, "pandas_gbq", falso)
    assert vd.main(["--subir"]) == 0
    (tabla, nombre), opciones = llamadas[0]
    assert nombre == "modelado.creditos_comparacion"
    assert opciones == {"project_id": "nb-rsk-scoring-dev", "if_exists": "replace",
                        "location": "us-central1"}
    assert len(tabla) == vd.TEST_ESPERADO + vd.TRAIN_ESPERADO


def test_main_rechaza_y_no_sube_si_la_tabla_falla(monkeypatch, capsys):
    llamadas = []
    monkeypatch.setattr(vd, "subir", lambda *a: llamadas.append(a))
    monkeypatch.setattr(vd, "TEST_ESPERADO", 1)
    assert vd.main(["--subir"]) == 1
    assert llamadas == []
    assert "RECHAZADO" in capsys.readouterr().out


# --- Tabla del pipeline propio ---------------------------------------------------


@pytest.fixture
def tabla_pipeline_real():
    config = vd.cargar_config()
    return vd.construir_tabla_pipeline(vd.leer_particion("estratificado_train", config),
                                       vd.leer_particion("estratificado_test", config))


def test_la_tabla_del_pipeline_conserva_las_particiones_procesadas(tabla_pipeline_real):
    config = vd.cargar_config()
    test = vd.leer_particion("estratificado_test", config)
    parte_test = tabla_pipeline_real[tabla_pipeline_real[vd.PARTICION] == "TEST"]
    assert list(tabla_pipeline_real.columns) == [vd.ID, *test.columns, vd.PARTICION]
    assert parte_test[vd.ID].iloc[0] == "te00000"
    assert parte_test.drop(columns=[vd.ID, vd.PARTICION]).reset_index(drop=True).equals(test)


def test_la_tabla_real_del_pipeline_cumple_el_contrato(tabla_pipeline_real):
    excluidas = vd.variables_excluidas()
    assert len(excluidas) == 8 and "saldo_mora" in excluidas
    assert vd.verificar_tabla_pipeline(tabla_pipeline_real, "Pago_atiempo", excluidas) == []


@pytest.mark.parametrize("defecto, mensaje", [
    ("fuga", "columnas con fuga"),
    ("particion", "particion no validos"),
    ("ids", "repetidos"),
    ("test", "el test no es el de Python"),
    ("train", "el train no es el de Python"),
])
def test_cada_defecto_de_la_tabla_del_pipeline_se_detecta(tabla_pipeline_real, defecto, mensaje):
    tabla = tabla_pipeline_real
    if defecto == "fuga":
        tabla = tabla.assign(saldo_mora=0)
    elif defecto == "particion":
        tabla.loc[0, vd.PARTICION] = "VALIDATE"
    elif defecto == "ids":
        tabla.loc[1, vd.ID] = tabla.loc[0, vd.ID]
    elif defecto == "test":
        tabla = tabla.iloc[:-1]
    else:
        tabla = tabla.iloc[1:]
    problemas = vd.verificar_tabla_pipeline(tabla, "Pago_atiempo", vd.variables_excluidas())
    assert any(mensaje in p for p in problemas)


def test_main_sube_la_tabla_del_pipeline_con_su_nombre(monkeypatch, capsys):
    llamadas = []
    falso = types.SimpleNamespace(to_gbq=lambda *a, **k: llamadas.append((a, k)))
    monkeypatch.setitem(sys.modules, "pandas_gbq", falso)
    assert vd.main(["--tabla", "pipeline", "--subir"]) == 0
    (tabla, nombre), _ = llamadas[0]
    assert nombre == "modelado.creditos_pipeline"
    assert len(tabla) == vd.TEST_ESPERADO + vd.TRAIN_ESPERADO
    assert "APROBADO" in capsys.readouterr().out
