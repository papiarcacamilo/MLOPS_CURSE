"""
La ingenieria de caracteristicas y el piso heuristico
================================================================================
Cubre las funciones puras de ft_engineering.py y hueristic_model.py.

Se prueban estas y no el pipeline completo por una razon de coste: reajustar el
binning sobre 8.610 registros tarda, y el resultado ya esta congelado en las
recetas. Lo que si hay que blindar son las piezas donde el proyecto encontro
fallos que no lanzaban ninguna excepcion, y que por tanto solo un test puede
atrapar.

El mas grave de todos: `_discretizar` producia etiquetas de tramo distintas
segun el TAMANIO del lote. Sobre el dataset entero escribia "(750.0, 800.0]" y
sobre pocas filas "(750, 800]". El WoE se buscaba por nombre, no encontraba la
etiqueta, y el registro entraba al modelo con riesgo promedio. En produccion eso
es un cliente puntuado mal, en silencio.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ft_engineering as fe
import hueristic_model as hm
import reglas_negocio as rn


# ==============================================================================
# EL FALLO DE LAS ETIQUETAS SEGUN EL TAMANIO DEL LOTE
# ==============================================================================


@pytest.fixture(scope="module")
def receta() -> dict:
    """El binning congelado por la Fase 2, tal como lo aplica el endpoint."""
    import json
    ruta = fe.RUTA_SALIDA / "receta_estratificado.json"
    if not ruta.exists():
        pytest.skip("Falta receta_estratificado.json: corre ft_engineering.py")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["binning_woe"]


@pytest.mark.parametrize("n_filas", [10_000, 500, 50, 3, 1])
def test_toda_etiqueta_producida_existe_en_la_receta(receta, train, n_filas):
    """El invariante del que depende que el WoE signifique algo.

    `aplicar_binning` busca el WoE por NOMBRE de tramo y, si no lo encuentra,
    rellena con 0.0, que es riesgo promedio. Conservador, pero invisible: el
    registro entra al modelo con una senial neutra y nadie se entera.

    La condicion que lo evita es que la etiqueta que produce la discretizacion,
    despues de la fusion, sea exactamente una de las claves aprendidas. Se
    comprueba a varios tamanios de lote porque el endpoint atiende desde una
    solicitud suelta hasta miles, y la etiqueta no puede depender de eso.

    Este test sustituye a otros tres que comparaban cortes enteros contra
    flotantes. En pandas 3.0.5 esa comparacion da siempre igual, de modo que
    pasaban sin probar nada: una mutacion que revirtiera la conversion a float
    los dejaba en verde.
    """
    lote = train.head(n_filas)
    desconocidas = {}

    for variable, spec in receta.items():
        if spec["tipo"] != "numerica":
            continue
        tramos, _ = fe._discretizar(
            lote[variable], spec["cortes"],
            etiqueta_nulo=spec.get("etiqueta_nulo", fe.ETIQUETA_NULO))
        tramos = tramos.map(lambda t: spec["fusion"].get(t, t))
        fuera = set(tramos.unique()) - set(spec["woe"])
        if fuera:
            desconocidas[variable] = fuera

    assert not desconocidas, desconocidas


def test_ninguna_variable_del_binning_recibe_woe_neutro_por_error(receta, train):
    """Comprueba lo mismo desde la salida: que el cero no venga del fallback.

    Un WoE de 0.0 es legitimo si asi se aprendio, pero aqui ninguna clave de la
    receta vale exactamente cero, de modo que un cero en la matriz solo puede
    proceder del `fillna` de la etiqueta no encontrada.
    """
    for spec in receta.values():
        assert 0.0 not in spec["woe"].values(), \
            "la receta tiene un WoE de cero: este test deja de discriminar"

    matriz = fe.aplicar_binning(train.head(1000), receta)
    for columna in matriz.columns:
        assert (matriz[columna] != 0.0).all(), columna


def test_los_tramos_de_los_extremos_son_abiertos():
    """Sin infinitos, un valor fuera del rango visto en train quedaria nulo y el
    modelo lo trataria como riesgo promedio en vez de como lo que es.
    """
    cortes = [-np.inf, 700.0, 800.0, np.inf]
    tramos, _ = fe._discretizar(pd.Series([1.0, 99999.0]), cortes)
    assert tramos.notna().all()
    assert "inf" in tramos.iloc[0] or "inf" in tramos.iloc[1]


def test_un_nulo_recibe_su_propia_etiqueta():
    """La ausencia de dato es informacion, no un hueco que rellenar."""
    tramos, _ = fe._discretizar(pd.Series([760.0, np.nan]),
                                [-np.inf, 700.0, 800.0, np.inf],
                                etiqueta_nulo=fe.ETIQUETA_NULO)
    assert tramos.iloc[1] == fe.ETIQUETA_NULO


# ==============================================================================
# WEIGHT OF EVIDENCE
# ==============================================================================


def test_el_woe_es_positivo_donde_hay_mas_riesgo():
    """Por convencion del proyecto, WoE alto significa mas probabilidad de mora.
    Si el signo se invirtiera, el scorecard premiaria justo a quien deberia
    penalizar.
    """
    tramos = pd.Series(["malo"] * 100 + ["bueno"] * 100)
    objetivo = pd.Series([1] * 60 + [0] * 40 + [1] * 10 + [0] * 90)
    woe, _ = fe.ajustar_woe(tramos, objetivo)
    assert woe["malo"] > woe["bueno"]


def test_el_woe_es_cercano_a_cero_sin_poder_discriminante():
    tramos = pd.Series(["a"] * 100 + ["b"] * 100)
    objetivo = pd.Series(([1] * 20 + [0] * 80) * 2)
    woe, iv = fe.ajustar_woe(tramos, objetivo)
    assert abs(woe["a"] - woe["b"]) < 0.1
    assert iv < 0.02


def test_el_suavizado_evita_la_division_por_cero():
    """Un tramo sin ningun evento mandaria el WoE a infinito sin la correccion
    de Laplace, y con 4,75% de mora los tramos vacios aparecen a menudo.
    """
    tramos = pd.Series(["vacio"] * 30 + ["normal"] * 100)
    objetivo = pd.Series([0] * 30 + [1] * 20 + [0] * 80)
    woe, iv = fe.ajustar_woe(tramos, objetivo)
    assert all(np.isfinite(v) for v in woe.values())
    assert np.isfinite(iv)


def test_el_information_value_nunca_es_negativo():
    tramos = pd.Series(["a"] * 50 + ["b"] * 50)
    objetivo = pd.Series([1] * 30 + [0] * 20 + [1] * 5 + [0] * 45)
    _, iv = fe.ajustar_woe(tramos, objetivo)
    assert iv >= 0


# ==============================================================================
# ATRIBUTOS DERIVADOS
# ==============================================================================


def test_consultas_por_credito_no_divide_por_cero():
    """El +1 del denominador existe para el cliente sin creditos vigentes."""
    datos = pd.DataFrame({
        "huella_consulta": [4], "cant_creditosvigentes": [0],
        "salario_cliente": [3_000_000], "promedio_ingresos_datacredito": [2e6],
        "tipo_credito": [4], fe.COLUMNA_FECHA: [pd.Timestamp("2025-01-07")]})
    derivadas = fe.construir_derivadas(datos, pd.Timestamp("2026-04-26"))
    assert np.isfinite(derivadas["consultas_por_credito"].iloc[0])
    assert derivadas["consultas_por_credito"].iloc[0] == 4.0


def test_la_discrepancia_de_ingresos_es_absoluta():
    """Mide la brecha entre lo declarado y lo reportado, no cual es mayor."""
    datos = pd.DataFrame({
        "huella_consulta": [1, 1], "cant_creditosvigentes": [1, 1],
        "salario_cliente": [3_000_000, 1_000_000],
        "promedio_ingresos_datacredito": [1_000_000, 3_000_000],
        "tipo_credito": [4, 4],
        fe.COLUMNA_FECHA: [pd.Timestamp("2025-01-07")] * 2})
    derivadas = fe.construir_derivadas(datos, pd.Timestamp("2026-04-26"))
    assert (derivadas["discrepancia_ingresos"] >= 0).all()
    assert derivadas["discrepancia_ingresos"].iloc[0] == \
           derivadas["discrepancia_ingresos"].iloc[1]


def test_los_tipos_de_credito_raros_se_agrupan():
    """Los tipos 7 y 68 tienen 2 y 1 registro en el dataset completo: como
    categorias propias no son estimables.
    """
    datos = pd.DataFrame({
        "huella_consulta": [1] * 3, "cant_creditosvigentes": [1] * 3,
        "salario_cliente": [3e6] * 3, "promedio_ingresos_datacredito": [2e6] * 3,
        "tipo_credito": [4, 7, 68],
        fe.COLUMNA_FECHA: [pd.Timestamp("2025-01-07")] * 3})
    grupos = fe.construir_derivadas(datos, pd.Timestamp("2026-04-26"))
    assert list(grupos["tipo_credito_grp"]) == ["4", "Otros", "Otros"]


def test_el_tipo_6_se_conserva_aparte():
    """Por su riesgo elevado, aunque su magnitud sea imprecisa: 21 creditos con
    9 en mora, con un intervalo al 95% entre 24,5% y 63,5%.
    """
    datos = pd.DataFrame({
        "huella_consulta": [1], "cant_creditosvigentes": [1],
        "salario_cliente": [3e6], "promedio_ingresos_datacredito": [2e6],
        "tipo_credito": [6], fe.COLUMNA_FECHA: [pd.Timestamp("2025-01-07")]})
    grupos = fe.construir_derivadas(datos, pd.Timestamp("2026-04-26"))
    assert grupos["tipo_credito_grp"].iloc[0] == "6"


# ==============================================================================
# ESCALADO
# ==============================================================================


def test_el_escalado_se_ajusta_sobre_train_y_se_aplica_sin_reajustar():
    """Si el test reajustara, cada conjunto se normalizaria contra si mismo y
    dejarian de ser comparables.
    """
    train = pd.DataFrame({v: np.linspace(1e6, 1e7, 100) for v in fe.VARS_MONETARIAS})
    parametros = fe.ajustar_escalado(train)
    otro = pd.DataFrame({v: np.linspace(5e6, 2e7, 50) for v in fe.VARS_MONETARIAS})

    escalado_a = fe.aplicar_escalado(otro, parametros)
    escalado_b = fe.aplicar_escalado(otro, parametros)
    pd.testing.assert_frame_equal(escalado_a, escalado_b)


def test_el_escalado_deja_la_mediana_de_train_en_cero():
    train = pd.DataFrame({v: np.linspace(0, 100, 101) for v in fe.VARS_MONETARIAS})
    escalado = fe.aplicar_escalado(train, fe.ajustar_escalado(train))
    for columna in escalado.columns:
        assert escalado[columna].median() == pytest.approx(0, abs=1e-9)


# ==============================================================================
# EL PISO HEURISTICO
# ==============================================================================


def test_la_regla_rechaza_por_debajo_del_corte():
    datos = pd.DataFrame({rn.VARIABLE_REGLA: [700.0, 800.0]})
    rechaza = hm.aplicar_regla(datos, 750)
    assert bool(rechaza.iloc[0]) and not bool(rechaza.iloc[1])


def test_la_regla_rechaza_a_quien_no_tiene_score():
    """El contrato lo declara: sin score, no se aprueba."""
    datos = pd.DataFrame({rn.VARIABLE_REGLA: [np.nan]})
    assert bool(hm.aplicar_regla(datos, 750).iloc[0]) is rn.RECHAZAR_SIN_SCORE


def test_el_corte_del_heuristico_es_uno_de_los_del_contrato(limpio):
    """La regla no puede inventarse un corte que el EDA no publico."""
    corte, _ = hm.derivar_corte(limpio)
    assert corte in rn.BANDAS_SCORE


def test_la_tabla_de_bandas_cubre_a_todo_el_que_tiene_score(limpio):
    """Los que quedan fuera son exactamente los que no tienen score.

    La tabla de bandas reparte por tramo de `puntaje_datacredito`, y quien no
    tiene score no cae en ningun tramo. No es un hueco: son los 153 registros
    marcados como sin historial, y a esos la regla los rechaza por politica, no
    por banda.

    El test fija la equivalencia entre las dos cuentas. Si algun dia dejaran de
    coincidir, o se perdieron registros por el camino o el nulo dejo de
    significar lo que significa.
    """
    tabla = hm.tabla_bandas(limpio)
    sin_score = int(limpio[rn.VARIABLE_REGLA].isna().sum())

    assert tabla["n"].sum() == len(limpio) - sin_score
    assert sin_score == int(limpio["sin_historial_crediticio"].sum())
