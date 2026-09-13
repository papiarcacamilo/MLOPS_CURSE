"""
El contrato del EDA
================================================================================
Cubre reglas_negocio.py, que es el modulo del que todo lo demas depende. Si aqui
cambia un corte, cambia el binning, cambia el modelo y cambia la decision del
endpoint, de modo que es el sitio donde una regresion hace mas danio.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reglas_negocio as rn


# ==============================================================================
# BANDAS Y CORTES
# ==============================================================================


def test_las_bandas_son_las_publicadas():
    """Los cortes del score son los que documento la Fase 1, en orden."""
    assert rn.BANDAS_SCORE == [280, 600, 700, 750, 800, 850, 950]
    assert rn.BANDAS_SCORE == sorted(rn.BANDAS_SCORE)


def test_no_hay_banda_fantasma():
    """El 650 no esta, y esa ausencia costo encontrarla.

    Una version anterior incluia un corte en 650 que ningun analisis respaldaba.
    Producia un tramo que el modelo puntuaba sin que nadie lo hubiera medido.
    """
    assert 650 not in rn.BANDAS_SCORE


def test_los_cortes_de_negocio_caen_dentro_del_rango_oficial():
    """Ningun corte puede quedar fuera de 150-950, el rango de DataCredito."""
    limites = rn.REGLAS_VALIDACION["puntaje_datacredito"]
    for corte in rn.CORTES_NEGOCIO["puntaje_datacredito"]:
        if corte in (-np.inf, np.inf):
            continue
        assert limites["min"] <= corte <= limites["max"], corte


def test_la_variable_de_la_regla_existe_en_las_reglas():
    """La variable sobre la que decide el heuristico debe estar validada."""
    assert rn.VARIABLE_REGLA in rn.REGLAS_VALIDACION


def test_la_politica_de_sin_score_esta_declarada():
    """El endpoint la aplica, asi que el contrato tiene que publicarla."""
    assert isinstance(rn.RECHAZAR_SIN_SCORE, bool)


# ==============================================================================
# VALIDACION
# ==============================================================================


def test_el_lote_limpio_solo_incumple_la_regla_que_seniala(limpio):
    """La salida de la Fase 1 cumple su contrato salvo una excepcion, conocida.

    Es circular a proposito: si algun dia deja de cumplirlo, o cambio la limpieza
    o cambio el contrato, y las dos cosas hay que enterarse.

    La unica violacion admitida es el techo de `total_otros_prestamos`, y no es
    un descuido. Esa regla se escribio para SENIALAR los 13 registros no
    verificables, no para excluirlos: la Fase 1 decidio marcarlos sin imputar,
    porque no hay forma de distinguir deuda corporativa legitima de un error de
    digitacion. Esos 13 entraron al entrenamiento con su bandera puesta, y por
    eso el endpoint tampoco los rechaza.

    El test fija esa excepcion en lugar de silenciarla: cualquier violacion nueva
    lo rompe.
    """
    violaciones = rn.validar_dataframe(limpio)
    esperada = "total_otros_prestamos: 13 valores sobre el maximo (1000000000)"
    assert violaciones == [esperada], violaciones


def test_los_registros_sospechosos_llevan_su_bandera(limpio):
    """Los 13 del techo estan marcados, que es el tratamiento que se les dio."""
    sobre_el_techo = limpio["total_otros_prestamos"] > 1_000_000_000
    assert sobre_el_techo.sum() == 13
    marcados = limpio.loc[sobre_el_techo, "total_otros_prestamos_sospechoso"]
    assert marcados.astype(str).str.lower().isin(["true", "1"]).all()


def test_detecta_un_valor_bajo_el_minimo():
    datos = pd.DataFrame({"edad_cliente": [42, 12]})
    reglas = {"edad_cliente": rn.REGLAS_VALIDACION["edad_cliente"]}
    violaciones = rn.validar_dataframe(datos, reglas)
    assert len(violaciones) == 1
    assert "bajo el minimo" in violaciones[0]


def test_detecta_un_valor_fuera_del_dominio():
    datos = pd.DataFrame({"tipo_credito": [4, 99]})
    reglas = {"tipo_credito": rn.REGLAS_VALIDACION["tipo_credito"]}
    violaciones = rn.validar_dataframe(datos, reglas)
    assert len(violaciones) == 1
    assert "dominio" in violaciones[0]


def test_detecta_una_columna_ausente():
    violaciones = rn.validar_dataframe(
        pd.DataFrame({"otra_cosa": [1]}),
        {"edad_cliente": rn.REGLAS_VALIDACION["edad_cliente"]})
    assert len(violaciones) == 1
    assert "ausente" in violaciones[0]


def test_los_nulos_permitidos_no_son_violacion():
    """El score nulo es ausencia de historial, no un error."""
    datos = pd.DataFrame({"puntaje_datacredito": [780.0, np.nan]})
    reglas = {"puntaje_datacredito": rn.REGLAS_VALIDACION["puntaje_datacredito"]}
    assert rn.validar_dataframe(datos, reglas) == []


def test_los_nulos_no_permitidos_si_lo_son():
    datos = pd.DataFrame({"edad_cliente": [42.0, np.nan]})
    reglas = {"edad_cliente": rn.REGLAS_VALIDACION["edad_cliente"]}
    violaciones = rn.validar_dataframe(datos, reglas)
    assert len(violaciones) == 1
    assert "nulos" in violaciones[0]


# ==============================================================================
# FECHAS: EL FALLO QUE COSTO 21 FECHAS FUTURAS
# ==============================================================================


def test_dayfirst_interpreta_el_formato_de_origen():
    """7/01/2025 es el 7 de enero, no el 1 de julio.

    Sin declarar la convencion, pandas elige mes primero en las fechas ambiguas
    y desplaza el rango del dataset de [2024-11-26, 2026-04-26] a
    [2024-01-12, 2026-12-02], inventando 21 fechas futuras que no existen.
    """
    fechas = rn._a_fecha(pd.Series(["7/01/2025 14:40"]), dayfirst=True)
    assert fechas.iloc[0].day == 7
    assert fechas.iloc[0].month == 1


def test_acepta_iso_sin_que_dayfirst_lo_estropee():
    """El dataset limpio ya viene en ISO y debe seguir leyendose igual."""
    fechas = rn._a_fecha(pd.Series(["2025-01-07 14:40:00"]), dayfirst=True)
    assert fechas.iloc[0].day == 7
    assert fechas.iloc[0].month == 1


def test_una_fecha_ilegible_queda_nula_sin_reventar():
    fechas = rn._a_fecha(pd.Series(["no es una fecha"]), dayfirst=True)
    assert fechas.isna().all()


def test_el_lote_limpio_no_tiene_fechas_futuras(limpio):
    """La comprobacion que atrapo el fallo original."""
    reglas = {"fecha_prestamo": rn.REGLAS_VALIDACION["fecha_prestamo"]}
    violaciones = rn.validar_dataframe(limpio, reglas)
    assert not any("futura" in v for v in violaciones), violaciones


# ==============================================================================
# COHERENCIA DEL CONTRATO CONSIGO MISMO
# ==============================================================================


@pytest.mark.parametrize("columna,regla", sorted(rn.REGLAS_VALIDACION.items()))
def test_cada_regla_declara_su_fuente(columna, regla):
    """Una regla sin procedencia no se puede defender ante un revisor."""
    assert regla.get("fuente"), columna


@pytest.mark.parametrize("columna,regla", sorted(rn.REGLAS_VALIDACION.items()))
def test_ningun_minimo_supera_a_su_maximo(columna, regla):
    if regla.get("min") is not None and regla.get("max") is not None:
        assert regla["min"] < regla["max"], columna
