"""
El motor de inferencia
================================================================================
Cubre model_deploy.py. Aqui vive la garantia central de la Fase 4: que un cliente
nuevo atraviese exactamente las mismas transformaciones que el entrenamiento.

Tres de estos tests los pidio danielCH26 en la revision de la PR4, y el cuarto
en la de la PR6:

    puntaje = 0                    -> bandera activada y score a nulo
    el scorecard reproduce el modelo
    el orden de la cadena          -> sanear antes de validar
    varias anomalias a la vez      -> que regla domina sobre cual

Los demas fijan los fallos que la auditoria encontro, para que no vuelvan.

NINGUN TEST ESCRIBE EN EL REGISTRO DE PRODUCCION. Todo lo que puntua lo hace con
`guardar_registro=False`.
================================================================================
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

import ft_engineering as fe
import model_deploy as md
import model_evaluation


# ==============================================================================
# SANEAMIENTO: REPRODUCIR LA FASE 1
# ==============================================================================


def test_sanear_reproduce_la_limpieza_de_la_fase_1():
    """El criterio de aceptacion del endpoint, sobre los 10.763 registros.

    Si difiere en un solo valor, el modelo esta recibiendo datos preparados de
    otra forma que aquellos con los que aprendio. No lanzaria ningun error:
    seguiria devolviendo probabilidades, solo que equivocadas.
    """
    v = md.verificar_saneamiento()
    assert v["reproduce_el_limpio"], v["diferencias"]
    assert v["registros"] == 10_763


def test_sanear_es_idempotente():
    """Sanear dos veces debe dar lo mismo que sanear una.

    No lo era. Sobre un lote ya saneado apagaba 103 banderas de edad, 187 de
    salario y 47 de tendencia, y rellenaba con la mediana los 105 nulos que
    marcaban ausencia de historial. Sin un error, con probabilidades distintas.
    """
    v = md.verificar_saneamiento()
    assert v["idempotente"], v["diferencias_segunda_pasada"]


def test_las_constantes_de_saneamiento_estan_congeladas():
    """Se estimaron una vez sobre el dataset completo, en la Fase 1.

    Recalcularlas sobre el lote entrante seria la forma mas facil de introducir
    train-serving skew sin que nada falle: una mediana sobre 50 solicitudes no
    es la mediana con la que el modelo aprendio.

    SI ESTE TEST FALLA, no se actualiza la cifra para que pase. Un cambio
    legitimo exige reestimar la constante sobre la base limpia, regenerar el
    artefacto con model_deploy.py y publicar una version nueva del modelo, porque
    el endpoint ya no reproduciria lo que se evaluo. Si nadie decidio ese cambio,
    es un bug.
    """
    c = md.SANEAMIENTO
    assert c["edad_umbral"] == 90
    assert c["edad_mediana"] == 42.0
    assert c["salario_mediana"] == 3_000_000.0
    assert c["puntaje_mediana"] == 791.0
    assert (c["score_min"], c["score_max"]) == (150, 950)


# ==============================================================================
# EL CASO QUE PIDIO DANIEL: puntaje = 0
# ==============================================================================


def test_un_puntaje_cero_activa_la_bandera_y_anula_el_score(solicitud):
    """Cero no es el peor score posible: es ausencia de historial.

    Tratarlo como el numero 0 haria que el modelo leyera a un cliente sin
    historial como si tuviera el peor score del mercado, que son situaciones
    distintas.
    """
    solicitud["puntaje_datacredito"] = 0
    saneado = md.sanear(pd.DataFrame([solicitud]))

    assert saneado["sin_historial_crediticio"].iloc[0] == 1
    assert pd.isna(saneado["puntaje_datacredito"].iloc[0])


@pytest.mark.parametrize("valor", [0, -5, 149, 951, 9999])
def test_cualquier_score_fuera_del_rango_oficial_pasa_a_nulo(solicitud, valor):
    solicitud["puntaje_datacredito"] = valor
    saneado = md.sanear(pd.DataFrame([solicitud]))
    assert saneado["sin_historial_crediticio"].iloc[0] == 1
    assert pd.isna(saneado["puntaje_datacredito"].iloc[0])


@pytest.mark.parametrize("valor", [150, 500, 780, 950])
def test_un_score_valido_se_conserva(solicitud, valor):
    solicitud["puntaje_datacredito"] = valor
    saneado = md.sanear(pd.DataFrame([solicitud]))
    assert saneado["sin_historial_crediticio"].iloc[0] == 0
    assert saneado["puntaje_datacredito"].iloc[0] == valor


def test_una_edad_imposible_se_corrige_y_deja_rastro(solicitud):
    solicitud["edad_cliente"] = 122
    saneado = md.sanear(pd.DataFrame([solicitud]))
    assert bool(saneado["edad_cliente_corregida"].iloc[0])
    assert saneado["edad_cliente"].iloc[0] == md.SANEAMIENTO["edad_mediana"]


def test_un_salario_cero_se_corrige_y_deja_rastro(solicitud):
    solicitud["salario_cliente"] = 0
    saneado = md.sanear(pd.DataFrame([solicitud]))
    assert bool(saneado["salario_cliente_corregido"].iloc[0])
    assert saneado["salario_cliente"].iloc[0] == md.SANEAMIENTO["salario_mediana"]


# ==============================================================================
# EL CASO QUE PIDIO DANIEL: EL ORDEN DE LA CADENA
# ==============================================================================


def test_sanear_va_antes_de_validar(solicitud):
    """Un score fuera de rango NO debe producir rechazo tecnico.

    Es la decision de disenio mas discutible de la Fase 4, y por eso se fija
    aqui. El contrato exige el score entre 150 y 950, pero la Fase 1 documento
    que salirse de ahi no es un error sino ausencia de historial, con tratamiento
    definido. Validar primero rechazaria justo lo que el EDA decidio conservar.
    """
    solicitud["puntaje_datacredito"] = 0
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["decision"].iloc[0] != md.DECISION_TECNICO


def test_lo_que_no_tiene_tratamiento_si_da_rechazo_tecnico(solicitud):
    """Una edad de 12 anios no tiene regla que aplicar: puntuar seria inventar."""
    solicitud["edad_cliente"] = 12
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["decision"].iloc[0] == md.DECISION_TECNICO
    assert "edad_cliente" in salida["motivo"].iloc[0]


def test_el_techo_de_otros_prestamos_no_es_rechazo_tecnico(solicitud):
    """El contrato lo puso para senialar, no para excluir.

    Los 13 registros por encima de 1.000 millones entraron al entrenamiento con
    su bandera puesta. Negarlos en produccion contradiria lo que se entreno.
    """
    solicitud["total_otros_prestamos"] = 2_000_000_000
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["decision"].iloc[0] != md.DECISION_TECNICO


def test_una_deuda_negativa_si_da_rechazo_tecnico(solicitud):
    """El minimo si se conserva: un monto negativo no tiene tratamiento."""
    solicitud["total_otros_prestamos"] = -5
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["decision"].iloc[0] == md.DECISION_TECNICO


# ==============================================================================
# EL CASO QUE PIDIO DANIEL EN LA PR6: VARIAS ANOMALIAS EN UNA SOLICITUD
# ==============================================================================


def test_dos_anomalias_con_tratamiento_activan_ambas_banderas(solicitud):
    """Sin score y con deuda sobre el techo: se tratan las dos y se puntua.

    Ninguna de las dos es motivo de rechazo tecnico, de modo que la solicitud
    llega al modelo con ambas banderas puestas, igual que en el entrenamiento.
    """
    solicitud["puntaje_datacredito"] = 0
    solicitud["total_otros_prestamos"] = 2_000_000_000
    datos = pd.DataFrame([solicitud])

    saneado = md.sanear(datos)
    assert bool(saneado["sin_historial_crediticio"].iloc[0])
    assert bool(saneado["total_otros_prestamos_sospechoso"].iloc[0])

    salida = md.predecir_lote(datos, origen="test", guardar_registro=False)
    assert salida["decision"].iloc[0] != md.DECISION_TECNICO
    assert not pd.isna(salida["probabilidad_mora"].iloc[0])


def test_una_anomalia_sin_tratamiento_domina_sobre_la_falta_de_score(solicitud):
    """El rechazo tecnico va antes que la politica de sin score.

    Una edad de 12 anios no tiene regla que aplicar, asi que no se puntua: la
    respuesta nombra la edad y no la ausencia de score.
    """
    solicitud["puntaje_datacredito"] = 0
    solicitud["edad_cliente"] = 12
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["decision"].iloc[0] == md.DECISION_TECNICO
    assert "edad_cliente" in salida["motivo"].iloc[0]
    assert pd.isna(salida["probabilidad_mora"].iloc[0])


def test_con_varias_violaciones_el_motivo_nombra_la_primera_del_contrato(solicitud):
    """Limitacion conocida: se reporta una violacion por solicitud.

    Es la primera segun el orden de REGLAS_ENDPOINT. Corregida esa, el endpoint
    senialaria la siguiente. Se fija aqui para que un cambio de este
    comportamiento sea deliberado y no un efecto secundario.
    """
    solicitud["edad_cliente"] = 12
    solicitud["plazo_meses"] = 0
    motivo = md.validar_filas(pd.DataFrame([solicitud])).iloc[0]

    orden = list(md.REGLAS_ENDPOINT)
    primera = min(("edad_cliente", "plazo_meses"), key=orden.index)
    segunda = max(("edad_cliente", "plazo_meses"), key=orden.index)
    assert motivo.startswith(primera)
    assert segunda not in motivo


# ==============================================================================
# ESQUEMA
# ==============================================================================


def test_un_lote_con_otro_esquema_falla_con_mensaje_claro():
    """Antes reventaba con KeyError dentro de sanear, que el endpoint traducia
    a un error 500: como si el servicio estuviera roto en vez de la peticion.
    """
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        md.predecir_lote(pd.DataFrame({"a": [1], "b": [2]}),
                         guardar_registro=False)


def test_el_mensaje_nombra_las_columnas_que_faltan(solicitud):
    incompleto = pd.DataFrame([solicitud]).drop(columns=["plazo_meses"])
    with pytest.raises(ValueError, match="plazo_meses"):
        md.predecir_lote(incompleto, guardar_registro=False)


def test_un_lote_vacio_falla():
    with pytest.raises(ValueError, match="no contiene solicitudes"):
        md.predecir_lote(pd.DataFrame(), guardar_registro=False)


def test_verificar_esquema_lista_lo_que_falta(solicitud):
    faltan = md.verificar_esquema(
        pd.DataFrame([solicitud]).drop(columns=["edad_cliente", "plazo_meses"]))
    assert set(faltan) == {"edad_cliente", "plazo_meses"}


# ==============================================================================
# LA CADENA DE INFERENCIA
# ==============================================================================


def test_partir_del_crudo_equivale_a_partir_del_limpio(cadena, crudo, limpio):
    """La prueba definitiva contra el train-serving skew.

    La ruta del endpoint arranca del archivo de origen; la que se uso para
    evaluar, del dataset ya limpio. Si no dan el mismo numero, lo que se midio no
    es lo que el servicio responde.
    """
    limpio = limpio.copy()
    limpio[fe.COLUMNA_FECHA] = pd.to_datetime(limpio[fe.COLUMNA_FECHA])

    p_endpoint = cadena.predict_proba(md.sanear(crudo))[:, 1]
    p_evaluado = cadena.predict_proba(limpio.drop(columns=[md.TARGET]))[:, 1]

    assert np.abs(p_endpoint - p_evaluado).max() == pytest.approx(0, abs=1e-12)


def test_la_cadena_no_reajusta_nada(cadena):
    """Reutiliza los pasos ya ajustados del joblib; no hay ningun fit."""
    ct = cadena.named_steps["caracteristicas"]
    assert hasattr(ct, "transformers_"), "el ColumnTransformer no venia ajustado"
    assert hasattr(ct.named_transformers_["woe"], "receta_")


def test_la_cadena_se_cachea_por_proceso():
    """Sin cache, cada peticion deserializaba el joblib: 213 ms por lote de 50
    frente a 96 ms ahora, con las mismas decisiones.
    """
    assert md.cadena_inferencia() is md.cadena_inferencia()


def test_el_modelo_tiene_las_17_caracteristicas(cadena):
    lr = cadena.named_steps["modelo"].named_steps["logisticregression"]
    assert lr.coef_.shape[1] == 17


# ==============================================================================
# DECISION
# ==============================================================================


def test_las_decisiones_son_las_tres_declaradas(lote):
    salida = md.predecir_lote(lote, origen="test", guardar_registro=False)
    permitidas = {md.DECISION_APROBAR, md.DECISION_RECHAZAR, md.DECISION_TECNICO}
    assert set(salida["decision"]).issubset(permitidas)


def test_ninguna_respuesta_deja_el_motivo_vacio(lote):
    """Un rechazo sin explicacion no se puede llevar a un comite de credito."""
    salida = md.predecir_lote(lote, origen="test", guardar_registro=False)
    assert (salida["motivo"].astype(str).str.len() > 0).all()


def test_el_umbral_del_artefacto_es_el_de_la_evaluacion(artefacto):
    """El punto de operacion no se elige en despliegue: llega fijado."""
    assert artefacto["umbral"]["valor"] == 0.0661
    assert "fuera de fold" in artefacto["umbral"]["fijado_en"]


def test_por_encima_del_umbral_se_rechaza(solicitud, umbral):
    """Un perfil claramente malo debe caer del lado del rechazo."""
    solicitud.update(tipo_credito=6, plazo_meses=72, puntaje_datacredito=610,
                     salario_cliente=1_200_000,
                     total_otros_prestamos=80_000_000,
                     tendencia_ingresos="Decreciente")
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["probabilidad_mora"].iloc[0] >= umbral
    assert salida["decision"].iloc[0] == md.DECISION_RECHAZAR


def test_un_rechazo_tecnico_no_trae_probabilidad(solicitud):
    solicitud["plazo_meses"] = 0
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)
    assert salida["decision"].iloc[0] == md.DECISION_TECNICO
    assert pd.isna(salida["probabilidad_mora"].iloc[0])
    assert salida["banda"].iloc[0] == "SIN_BANDA"


def test_la_politica_de_sin_score_se_declara_en_el_motivo(solicitud, umbral):
    """Una probabilidad baja junto a un rechazo parece un error si no se explica.

    El perfil esta elegido para que el modelo lo puntue MUY por debajo del
    umbral (0,0123 frente a 0,0661): si aun asi se rechaza, es la politica del
    contrato y no el modelo, y el motivo tiene que decirlo.
    """
    solicitud.update(puntaje_datacredito=0, plazo_meses=12,
                     capital_prestado=3_000_000, salario_cliente=9_000_000,
                     total_otros_prestamos=1_000_000, cuota_pactada=280_000,
                     tendencia_ingresos="Creciente", edad_cliente=55,
                     cant_creditosvigentes=0, huella_consulta=0,
                     tipo_credito=10, promedio_ingresos_datacredito=8_000_000)
    salida = md.predecir_lote(pd.DataFrame([solicitud]), origen="test",
                              guardar_registro=False)

    assert salida["probabilidad_mora"].iloc[0] < umbral, \
        "el perfil dejo de ser de bajo riesgo: el test ya no prueba la politica"
    assert salida["decision"].iloc[0] == md.DECISION_RECHAZAR
    assert md.MOTIVO_POLITICA in salida["motivo"].iloc[0]


# ==============================================================================
# EL CASO QUE PIDIO DANIEL: EL SCORECARD REPRODUCE EL MODELO
# ==============================================================================


def test_el_scorecard_reproduce_la_prediccion_del_modelo(scorecard):
    """Si no lo hiciera, la tabla que ve el analista no representa lo que decide
    el sistema, y el scorecard seria decorativo.
    """
    v = model_evaluation.verificar_scorecard(
        joblib.load(md.RUTA_MODELO), _muestra_de_test(), scorecard)
    assert v["coincide"], v["error_max_probabilidad"]
    assert v["error_max_probabilidad"] < 1e-9


def _muestra_de_test() -> pd.DataFrame:
    datos = pd.read_csv(fe.RUTA_SALIDA / "estratificado_test.csv",
                        sep=fe.SEPARADOR, encoding=fe.ENCODING)
    return datos.drop(columns=[md.TARGET]).head(200)


def test_el_puntaje_invierte_el_sentido_de_la_probabilidad(scorecard):
    """Mas puntos significan menos riesgo. Si se invirtiera, un analista
    aprobaria justo a quien deberia negar.
    """
    escala = scorecard["escala"]
    riesgo_bajo = md.puntaje_desde_probabilidad(np.array([0.01]), escala)[0]
    riesgo_alto = md.puntaje_desde_probabilidad(np.array([0.50]), escala)[0]
    assert riesgo_bajo > riesgo_alto


def test_las_bandas_ordenan_el_riesgo(artefacto):
    """De A a E la mora debe crecer de forma monotona."""
    tasas = [b["tasa_mora_train"] for b in artefacto["bandas"]]
    assert tasas == sorted(tasas, reverse=True), tasas


def test_las_bandas_cubren_toda_la_recta(artefacto):
    bandas = artefacto["bandas"]
    assert bandas[0]["puntaje_desde"] is None
    assert bandas[-1]["puntaje_hasta"] is None


# ==============================================================================
# EL ARTEFACTO DE SERVICIO
# ==============================================================================


def test_el_artefacto_declara_el_esquema_de_entrada(artefacto):
    esquema = artefacto["esquema_entrada"]
    assert esquema["columnas_requeridas"] == md.COLUMNAS_REQUERIDAS
    assert set(esquema["nulos_admitidos"]) == set(md.NULOS_ADMITIDOS)


def test_el_esquema_no_pide_variables_con_fuga():
    """Los saldos son posteriores al desembolso: en originacion no existen."""
    for fuga in fe.VARIABLES_EXCLUIDAS:
        assert fuga not in md.COLUMNAS_REQUERIDAS


def test_el_esquema_no_pide_el_objetivo():
    assert md.TARGET not in md.COLUMNAS_REQUERIDAS


def test_la_version_del_modelo_identifica_el_archivo():
    """El hash viaja en cada fila del registro: sin el no se puede separar
    'cambio la poblacion' de 'alguien desplego otra version'.
    """
    version = md.version_modelo()
    assert len(version) == 12
    assert all(c in "0123456789abcdef" for c in version)
