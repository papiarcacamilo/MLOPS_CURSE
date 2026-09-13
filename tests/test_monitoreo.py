"""
El monitoreo y la medida de deriva
================================================================================
Cubre model_monitoring.py. Dos grupos de tests con propositos distintos:

    los que verifican el INSTRUMENTO   que el PSI cumple sus propiedades
                                       matematicas y coincide con una
                                       implementacion de referencia
    los que fijan las DECISIONES       que la linea base no se reconstruye, que
                                       el muestreo deduplica, que el control
                                       discrimina

Un medidor que nadie ha medido no sirve para decidir nada, y eso vale igual para
el PSI que para el modelo.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ft_engineering as fe
import model_deploy as md
import model_monitoring as mo


@pytest.fixture(scope="module")
def base() -> dict:
    if not mo.RUTA_LINEA_BASE.exists():
        pytest.skip("Falta linea_base_monitoreo.json: corre model_monitoring.py")
    return mo.cargar_linea_base()[0]


# ==============================================================================
# EL INSTRUMENTO: PROPIEDADES DEL PSI
# ==============================================================================


def test_psi_de_una_distribucion_contra_si_misma_es_cero():
    """La propiedad mas basica. Si fallara, cualquier lectura seria ruido."""
    d = {"a": 0.2, "b": 0.5, "c": 0.3}
    assert mo.psi(d, d) == pytest.approx(0, abs=1e-9)


def test_psi_es_simetrico():
    a = {"x": 0.7, "y": 0.3}
    b = {"x": 0.4, "y": 0.6}
    assert mo.psi(a, b) == pytest.approx(mo.psi(b, a), abs=1e-9)


def test_psi_nunca_es_negativo():
    a = {"x": 0.9, "y": 0.1}
    b = {"x": 0.1, "y": 0.9}
    assert mo.psi(a, b) >= 0


def test_psi_crece_cuando_la_distribucion_se_aleja():
    """Un indice que no ordenara las diferencias no serviria para vigilar."""
    base = {"x": 0.5, "y": 0.5}
    cerca = mo.psi(base, {"x": 0.55, "y": 0.45})
    lejos = mo.psi(base, {"x": 0.90, "y": 0.10})
    assert lejos > cerca > 0


def test_psi_coincide_con_una_implementacion_de_referencia():
    """Escrita desde la formula, sin reutilizar nada del modulo.

        PSI = suma( (obs - esp) * ln(obs / esp) )
    """
    def referencia(esp, obs, eps=mo.EPSILON):
        claves = sorted(set(esp) | set(obs))
        e = np.array([esp.get(k, 0.0) + eps for k in claves]); e /= e.sum()
        o = np.array([obs.get(k, 0.0) + eps for k in claves]); o /= o.sum()
        return float(((o - e) * np.log(o / e)).sum())

    esperado = {"a": 0.30, "b": 0.45, "c": 0.25}
    observado = {"a": 0.10, "b": 0.60, "c": 0.30}
    assert mo.psi(esperado, observado) == pytest.approx(
        referencia(esperado, observado), abs=1e-12)


def test_un_tramo_vacio_no_manda_el_psi_a_infinito():
    """Sin la correccion por epsilon, el indice se dispara en cuanto un tramo se
    queda sin observaciones, que con muestras pequenias pasa a menudo.
    """
    valor = mo.psi({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0})
    assert np.isfinite(valor)


def test_los_tramos_se_aparean_por_nombre_no_por_posicion():
    """Si una cohorte no tiene algun tramo, ese entra en cero en vez de
    desalinear todo el vector y comparar peras con manzanas.
    """
    completo = {"bajo": 0.3, "medio": 0.4, "alto": 0.3}
    parcial = {"alto": 0.6, "bajo": 0.4}
    assert np.isfinite(mo.psi(completo, parcial))


# ==============================================================================
# EL SEMAFORO
# ==============================================================================


@pytest.mark.parametrize("valor,estado", [
    (0.0,   mo.ESTADO_ESTABLE),
    (0.099, mo.ESTADO_ESTABLE),
    (0.10,  mo.ESTADO_MODERADO),
    (0.249, mo.ESTADO_MODERADO),
    (0.25,  mo.ESTADO_IMPORTANTE),
    (1.0,   mo.ESTADO_IMPORTANTE),
])
def test_el_semaforo_usa_los_umbrales_del_sector(valor, estado):
    """0,10 y 0,25 son la convencion en scorecards de credito, no una eleccion
    de este proyecto. Cambiarlos cambiaria todos los veredictos publicados.
    """
    assert mo.clasificar(valor) == estado


def test_los_umbrales_no_se_han_movido():
    assert mo.UMBRAL_PSI_MODERADO == 0.10
    assert mo.UMBRAL_PSI_IMPORTANTE == 0.25


# ==============================================================================
# LA LINEA BASE SE CONGELA
# ==============================================================================


def test_la_linea_base_se_lee_y_no_se_reconstruye():
    """Se reconstruia en cada corrida, que es exactamente el error contra el que
    advierte el propio docstring: si la referencia se recalcula sobre unos datos
    que alguien regenero, se desplaza en silencio junto con lo medido y una
    deriva lenta no se detecta jamas.
    """
    _, construida = mo.cargar_linea_base()
    assert not construida, "la linea base se reconstruyo en lugar de leerse"


def test_reconstruirla_es_explicito(monkeypatch):
    llamadas = []
    monkeypatch.setattr(mo, "construir_linea_base",
                        lambda: llamadas.append(1) or {"n": 0})
    monkeypatch.setattr(mo, "guardar_linea_base", lambda b: None)
    mo.cargar_linea_base(reconstruir=True)
    assert llamadas == [1]


def test_la_linea_base_describe_las_13_variables_y_las_4_banderas(base):
    assert len(base["variables"]) == 13
    assert len(base["banderas"]) == 4
    assert set(base["banderas"]) == set(fe.VARS_BINARIAS)


def test_cada_distribucion_de_la_linea_base_suma_uno(base):
    for nombre, detalle in base["variables"].items():
        total = sum(detalle["distribucion"].values())
        assert total == pytest.approx(1.0, abs=1e-4), nombre


def test_los_cortes_vienen_de_la_receta_del_modelo(base):
    """Con otros tramos, el indice mediria una cosa y el modelo veria otra."""
    receta = mo._tramos_de_la_receta()
    for variable, detalle in receta.items():
        if detalle["tipo"] == "numerica":
            assert base["variables"][variable]["cortes"] == detalle["cortes"]


def test_la_linea_base_declara_el_umbral_de_operacion(base, umbral):
    assert base["prediccion"]["umbral"] == umbral


# ==============================================================================
# PREPARAR: EL FALLO QUE HIZO CAER AL CONTROL
# ==============================================================================


def test_preparar_no_vuelve_a_sanear_lo_ya_saneado(train):
    """La primera version saneaba dos veces y apagaba las cuatro banderas, lo
    que llevaba el PSI del control a 0,1153 sobre su propia poblacion de ajuste.
    """
    preparada = mo.preparar(train)
    for bandera in fe.VARS_BINARIAS:
        antes = train[bandera].astype(str).str.lower().isin(["true", "1"]).sum()
        despues = preparada[bandera].astype(str).str.lower().isin(
            ["true", "1"]).sum()
        assert antes == despues, bandera


def test_preparar_conserva_los_nulos_intencionales_del_score(train):
    """Rellenarlos convertiria un 'no tiene historial' en un score promedio."""
    assert (mo.preparar(train)["puntaje_datacredito"].isna().sum()
            == train["puntaje_datacredito"].isna().sum())


def test_preparar_si_sanea_lo_que_viene_crudo(crudo):
    """El registro guarda la entrada tal como llego, y hay que rehacer el camino."""
    preparada = mo.preparar(crudo.head(500))
    for bandera in fe.VARS_BINARIAS:
        assert bandera in preparada.columns


def test_preparar_construye_las_derivadas(train):
    preparada = mo.preparar(train)
    for derivada in ("consultas_por_credito", "discrepancia_ingresos",
                     "tipo_credito_grp"):
        assert derivada in preparada.columns


# ==============================================================================
# MUESTREO
# ==============================================================================


def test_el_muestreo_cuenta_solicitudes_distintas_no_decisiones():
    """Una misma solicitud puntuada dos veces es UNA observacion de la
    poblacion. Sin esto, 250 decisiones sobre 50 solicitudes reales daban un PSI
    de prediccion de 0,2858 sobre una poblacion que no habia cambiado.
    """
    fila = {c: 1 for c in md.COLUMNAS_REQUERIDAS}
    fila["fecha_prestamo"] = "7/01/2025 14:40"
    repetido = pd.DataFrame([fila] * 10)
    repetido["momento"] = pd.Timestamp("2026-01-05")

    periodos = mo.muestrear(repetido)
    periodo = next(iter(periodos.values()))
    assert periodo["n_decisiones"] == 10
    assert periodo["n_solicitudes"] == 1


def test_un_periodo_con_poco_volumen_no_se_publica():
    """Es preferible un hueco declarado a un numero que nadie puede interpretar."""
    filas = []
    for i in range(5):
        fila = {c: i for c in md.COLUMNAS_REQUERIDAS}
        fila["fecha_prestamo"] = "7/01/2025 14:40"
        filas.append(fila)
    datos = pd.DataFrame(filas)
    datos["momento"] = pd.Timestamp("2026-01-05")

    periodo = next(iter(mo.muestrear(datos).values()))
    assert not periodo["suficiente"]


def test_el_muestreo_admite_los_dos_ejes():
    """Llegada responde si lo que entra hoy se parece; cosecha, como evoluciona
    el riesgo por vintage. Son preguntas distintas.
    """
    fila = {c: 1 for c in md.COLUMNAS_REQUERIDAS}
    datos = pd.DataFrame([{**fila, "fecha_prestamo": f"0{d}/01/2025 10:00"}
                          for d in (1, 2)])
    datos["momento"] = pd.Timestamp("2026-01-05")

    assert len(mo.muestrear(datos, "momento", "W")) == 1
    assert len(mo.muestrear(datos, "fecha_prestamo", "MS")) == 1


def test_las_fechas_del_registro_se_leen_con_dayfirst():
    """El registro guarda `fecha_prestamo` en el formato de origen d/m/Y. Dejar
    que pandas lo adivine desplaza el rango entero, que es el error que la
    Fase 1 ya pago una vez.
    """
    fila = {c: 1 for c in md.COLUMNAS_REQUERIDAS}
    fila["fecha_prestamo"] = "07/01/2025 10:00"
    datos = pd.DataFrame([fila])
    datos["momento"] = pd.Timestamp("2026-01-05")

    etiqueta = next(iter(mo.muestrear(datos, "fecha_prestamo", "MS")))
    assert etiqueta.startswith("2025-01")


# ==============================================================================
# EVALUACION DE UNA COHORTE
# ==============================================================================


def test_evaluar_exige_las_probabilidades(base):
    """El parametro era opcional y estaba roto: sobre datos del registro fallaba
    con un TypeError. Un parametro opcional que no funciona es peor que uno
    obligatorio.

    Y hay una razon de fondo: el monitoreo mide los pronosticos que el endpoint
    ENTREGO, no los que emitiria ahora.
    """
    with pytest.raises(ValueError, match="probabilidades"):
        mo.evaluar_cohorte("x", pd.DataFrame({"a": [1, 2]}), base, None)


def test_evaluar_exige_que_las_longitudes_cuadren(base):
    with pytest.raises(ValueError):
        mo.evaluar_cohorte("x", pd.DataFrame({"a": [1, 2, 3]}), base,
                           np.array([0.1, 0.2]))


def test_el_control_se_mantiene_por_debajo_del_umbral(base, train):
    """Sale de la MISMA poblacion sobre la que el modelo se ajusto, asi que debe
    dar cerca de cero. Si no lo diera, la medida estaria rota y el numero de la
    otra cohorte no merecerian credito.
    """
    muestra = train.sample(1500, random_state=fe.SEMILLA)
    salida = md.predecir_lote(muestra[md.COLUMNAS_REQUERIDAS],
                              origen="test", guardar_registro=False)
    resultado = mo.evaluar_cohorte(
        "control", muestra, base,
        salida["probabilidad_mora"].to_numpy(dtype=float))

    assert resultado["covariables"]["psi_maximo"] < mo.UMBRAL_PSI_MODERADO
    assert resultado["veredicto"] == "SIN CAMBIO RELEVANTE"


def test_una_cohorte_posterior_si_muestra_deriva(base):
    """temporal_test son creditos desembolsados desde 2025-07-12: una poblacion
    genuinamente distinta, no una simulacion.
    """
    te = pd.read_csv(fe.RUTA_SALIDA / "temporal_test.csv",
                     sep=fe.SEPARADOR, encoding=fe.ENCODING)
    te[fe.COLUMNA_FECHA] = pd.to_datetime(te[fe.COLUMNA_FECHA])
    salida = md.predecir_lote(te[md.COLUMNAS_REQUERIDAS], origen="test",
                              guardar_registro=False)
    resultado = mo.evaluar_cohorte(
        "reciente", te, base, salida["probabilidad_mora"].to_numpy(dtype=float))

    assert resultado["covariables"]["psi_maximo"] > mo.UMBRAL_PSI_MODERADO
    assert resultado["covariables"]["tabla"][0]["variable"] == "plazo_meses"


def test_la_evaluacion_devuelve_las_cuatro_seniales(base, train):
    muestra = train.sample(300, random_state=fe.SEMILLA)
    salida = md.predecir_lote(muestra[md.COLUMNAS_REQUERIDAS], origen="test",
                              guardar_registro=False)
    resultado = mo.evaluar_cohorte(
        "x", muestra, base, salida["probabilidad_mora"].to_numpy(dtype=float))

    for senial in ("covariables", "prediccion", "corte", "concepto"):
        assert senial in resultado
    assert resultado["veredicto"]


# ==============================================================================
# LA TRAMPA DE LA MADUREZ
# ==============================================================================


def test_el_concepto_reporta_los_tres_subgrupos(base):
    """El filtro de madurez no aisla el censurado: esta confundido con el plazo,
    de modo que selecciona creditos cortos y menos riesgosos. Publicar solo el
    subgrupo vencido reportaria el desempenio sobre la mitad mas corta.
    """
    te = pd.read_csv(fe.RUTA_SALIDA / "temporal_test.csv",
                     sep=fe.SEPARADOR, encoding=fe.ENCODING)
    te[fe.COLUMNA_FECHA] = pd.to_datetime(te[fe.COLUMNA_FECHA])
    salida = md.predecir_lote(te[md.COLUMNAS_REQUERIDAS], origen="test",
                              guardar_registro=False)
    concepto = mo.deriva_concepto(
        te, salida["probabilidad_mora"].to_numpy(dtype=float), base)

    assert concepto["medible"]
    for clave in ("cohorte_completa", "solo_vencidos", "solo_jovenes"):
        assert clave in concepto
    assert "sesgo_de_madurez" in concepto


def test_los_jovenes_tienen_mas_mora_que_los_vencidos(base):
    """Lo contrario de lo que el proyecto venia afirmando, y lo que obliga a
    reportar ambos subgrupos.
    """
    te = pd.read_csv(fe.RUTA_SALIDA / "temporal_test.csv",
                     sep=fe.SEPARADOR, encoding=fe.ENCODING)
    te[fe.COLUMNA_FECHA] = pd.to_datetime(te[fe.COLUMNA_FECHA])
    salida = md.predecir_lote(te[md.COLUMNAS_REQUERIDAS], origen="test",
                              guardar_registro=False)
    c = mo.deriva_concepto(te, salida["probabilidad_mora"].to_numpy(dtype=float),
                           base)

    assert c["solo_jovenes"]["tasa_mora"] > c["solo_vencidos"]["tasa_mora"]
    assert c["solo_jovenes"]["plazo_medio"] > c["solo_vencidos"]["plazo_medio"]


def test_el_concepto_no_se_mide_sin_la_etiqueta(base):
    sin_objetivo = pd.DataFrame({c: [1] * 200 for c in md.COLUMNAS_REQUERIDAS})
    c = mo.deriva_concepto(sin_objetivo, np.full(200, 0.05), base)
    assert not c["medible"]
    assert "objetivo" in c["motivo"]
