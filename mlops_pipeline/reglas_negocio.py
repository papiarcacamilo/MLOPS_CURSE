"""
Contrato de negocio derivado del EDA
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Origen   : Fase 1 - transformacion_eda.ipynb

QUE ES ESTE ARCHIVO

Es la fuente unica de los cortes, bandas y reglas de validacion que el EDA
establecio. Todo lo que viene despues -- el modelo heuristico, la ingenieria de
caracteristicas, el despliegue y el monitoreo -- APLICA este contrato. Ninguno
de ellos lo redefine.

POR QUE EXISTE

Hasta ahora las bandas de `puntaje_datacredito` vivian duplicadas en tres
lugares: la celda 63 del notebook de EDA, la constante CORTES_INICIALES de
ft_engineering.py, y la constante BANDAS_SCORE del modelo heuristico. Las dos
primeras coincidian; la tercera habia derivado por su cuenta e incluia un corte
en 650 que no proviene de ninguna parte y que generaba una banda de 8 registros
-- justo el tipo de tramo que la Fase 2 fusiona por inestable. El comentario que
lo acompaniaba afirmaba haber heredado las bandas de la Fase 2, cuando la Fase 2
hizo lo contrario.

El efecto sobre las metricas era nulo (la derivacion del corte ignora tramos con
menos del 1% de la cartera), pero la cadena EDA -> reglas -> features quedaba
rota. En un proyecto que se sostiene sobre su metodologia, la trazabilidad es
parte del producto.

DIRECCION DE LA DEPENDENCIA

    EDA (deriva)
      |
      v
    reglas_negocio.py  (publica el contrato)
      |
      +--> heuristic_model.py   decide sin modelo
      +--> ft_engineering.py    discretiza y transforma
      +--> model_deploy.py      valida clientes nuevos
      +--> model_monitoring.py  linea base de deriva

El heuristico NO depende de la ingenieria de caracteristicas. Solo necesita el
puntaje crudo y el contrato. Que antes dependiera de ella fue un accidente de
implementacion, no una necesidad.
================================================================================
"""

from __future__ import annotations

import pandas as pd


# ==============================================================================
# BANDAS DE RIESGO
# ==============================================================================

# Bandas de `puntaje_datacredito`, derivadas en la celda 63 del EDA.
#
# No son cuantiles: son cortes con significado comercial en el sistema
# financiero colombiano, sobre el rango oficial de DataCredito Experian
# [150, 950]. El EDA midio sobre ellas un gradiente de 64% de mora por debajo de
# 600 puntos a 2.91% por encima de 850, con un Information Value de 0.2136 -- el
# mas alto del dataset -- mientras su correlacion de Pearson (0.1212) se leeria
# aisladamente como "relacion muy debil". Esa distancia entre ambas medidas es el
# hallazgo metodologico central de la Fase 1.
BANDAS_SCORE = [280, 600, 700, 750, 800, 850, 950]

# Cortes de negocio por variable. La fusion automatica de la Fase 2 los ajusta si
# algun tramo resulta demasiado pequenio, pero el punto de partida se decide aqui.
CORTES_NEGOCIO = {
    "puntaje_datacredito": BANDAS_SCORE,

    # P3 | Cortes revisados en la Fase 2 sobre la evidencia de la Fase 1.
    # Con los iniciales ([0,6,12,18,24,36,90]) el tramo (12,18] tenia 275
    # registros y 12 eventos, por debajo de ambos minimos, y la fusion lo unia
    # con (18,24] -- que es justo donde empieza la senial (8.16% de mora). El
    # resultado diluia el gradiente y el IV de la variable BAJABA al
    # transformarla (0.1069 -> 0.0904 en la particion estratificada). La Fase 1
    # muestra que (12,18] (4.36%) se parece mucho mas a (6,12] (3.91%) que a
    # (18,24] (8.16%), asi que se agrupan desde el corte inicial y se deja
    # (18,24] aislado.
    "plazo_meses": [0, 6, 18, 24, 90],
}


# ==============================================================================
# REGLA HEURISTICA DE ORIGINACION
# ==============================================================================

# Variable sobre la que se construye la regla. Es la unica eleccion razonable:
# la Fase 1 la identifico como el predictor individual mas fuerte y es un dato
# que la entidad ya tiene sin coste adicional.
VARIABLE_REGLA = "puntaje_datacredito"

# Principio que define el corte. NO se optimiza ninguna metrica: hacerlo
# ajustaria la regla a train y dejaria de ser un piso de comparacion honesto
# frente a los modelos de aprendizaje automatico.
PRINCIPIO_CORTE = (
    "Rechazar donde el riesgo observado supera el promedio de la cartera. "
    "Recorriendo las bandas de score de mayor a menor, el corte es el limite "
    "superior de la primera banda cuya tasa de mora ya excede la tasa base."
)

# Las bandas con menos de este porcentaje de la cartera no participan en la
# derivacion del corte: con tan pocos registros su tasa observada es ruido.
MIN_POBLACION_BANDA_PCT = 1.0

# Los solicitantes sin score valido se rechazan por el mismo principio, no por
# comodidad: la Fase 2 midio su WoE en +0.4174 y su tasa observada supera la
# base de la cartera. Aprobarlos contradiria el criterio que define el corte.
RECHAZAR_SIN_SCORE = True


# ==============================================================================
# REGLAS DE VALIDACION
# ==============================================================================

# Derivadas del DICCIONARIO DE DATOS y de fuentes oficiales (celda 84 del EDA).
# Se aplican a los datos de entrenamiento y, sobre todo, a los clientes nuevos
# que lleguen al endpoint en la Fase 4: son el filtro que impide puntuar un
# registro imposible.
#
# CORRECCION DOCUMENTADA EN LA FASE 1: el rango de `puntaje_datacredito` era
# [0, 1000], un rango inventado sin fuente. El oficial de DataCredito Experian
# es [150, 950]. La regla anterior detectaba 1 registro invalido; la corregida
# detecta 153.
REGLAS_VALIDACION = {
    'edad_cliente':          {'min': 18,   'max': 90,         'nulos_permitidos': False,
                              'fuente': 'Mayoria de edad legal + brecha observada en la distribucion (70-120)'},
    'salario_cliente':       {'min': 1,    'max': 20_000_000, 'nulos_permitidos': False,
                              'fuente': 'Limite IQR sobre escala logaritmica (~18.79M), redondeado'},
    'capital_prestado':      {'min': 1,    'max': 50_000_000, 'nulos_permitidos': False,
                              'fuente': 'Maximo observado 41.4M + margen'},
    'plazo_meses':           {'min': 1,    'max': 120,        'nulos_permitidos': False,
                              'fuente': 'Maximo observado 90 meses + margen comercial'},
    'cuota_pactada':         {'min': 1,    'max': 5_000_000,  'nulos_permitidos': False,
                              'fuente': 'Rango observado'},
    'puntaje_datacredito':   {'min': 150,  'max': 950,        'nulos_permitidos': True,
                              'fuente': 'Rango oficial DataCredito Experian (Terminos y Condiciones '
                                        'Midatacredito). Nulos permitidos: representan ausencia de '
                                        'historial (ver seccion 2.4 del EDA)'},
    'cant_creditosvigentes': {'min': 0,    'max': 100,        'nulos_permitidos': False,
                              'fuente': 'Maximo observado 62 + margen'},
    'huella_consulta':       {'min': 0,    'max': 50,         'nulos_permitidos': False,
                              'fuente': 'Maximo observado 30 + margen'},
    'saldo_mora':            {'min': 0,    'max': None,       'nulos_permitidos': False,
                              'fuente': 'Un saldo no puede ser negativo'},
    'saldo_total':           {'min': 0,    'max': None,       'nulos_permitidos': False,
                              'fuente': 'Un saldo no puede ser negativo'},
    'saldo_principal':       {'min': 0,    'max': None,       'nulos_permitidos': False,
                              'fuente': 'Un saldo no puede ser negativo'},
    'saldo_mora_codeudor':   {'min': 0,    'max': None,       'nulos_permitidos': False,
                              'fuente': 'Un saldo no puede ser negativo'},
    'tipo_credito':          {'valores': [4, 6, 7, 9, 10, 68], 'nulos_permitidos': False,
                              'fuente': 'Codigos observados en el dataset (significado sin confirmar)'},
    'tipo_laboral':          {'valores': ['Empleado', 'Independiente'], 'nulos_permitidos': False,
                              'fuente': 'Dominio cerrado observado'},
    'tendencia_ingresos':    {'valores': ['Creciente', 'Decreciente', 'Estable', 'Sin_dato'],
                              'nulos_permitidos': False,
                              'fuente': 'Dominio cerrado tras reconstruccion (seccion 2.4 del EDA)'},
    'Pago_atiempo':          {'valores': [0, 1], 'nulos_permitidos': False,
                              'fuente': 'Variable objetivo binaria'},
    'total_otros_prestamos': {'min': 0,    'max': 1_000_000_000, 'nulos_permitidos': False,
                              'fuente': 'Un monto de deuda no puede ser negativo. Techo de 1.000M: '
                                        'por encima hay 13 registros no verificables (seccion 4.1); '
                                        'la regla los seniala en vez de dejarlos pasar'},
    'promedio_ingresos_datacredito': {'min': 0, 'max': 50_000_000, 'nulos_permitidos': True,
                              'fuente': 'Maximo observado 38.1M + margen. Nulos permitidos: 27% de '
                                        'ausencia sin causa estructural, conservada como informacion '
                                        '(seccion 2.2 del EDA)'},
    'creditos_sectorFinanciero': {'min': 0, 'max': 100, 'nulos_permitidos': False,
                              'fuente': 'Maximo observado 51 + margen'},
    'creditos_sectorCooperativo': {'min': 0, 'max': 50, 'nulos_permitidos': False,
                              'fuente': 'Maximo observado 13 + margen'},
    'creditos_sectorReal':   {'min': 0,    'max': 100,        'nulos_permitidos': False,
                              'fuente': 'Maximo observado 25 + margen'},
    # `dayfirst` es obligatorio aqui: el archivo de origen trae las fechas en
    # formato d/m/Y ("7/01/2025 14:40" es el 7 de enero). Sin declararlo, pandas
    # interpreta como mes el primer campo de las fechas ambiguas y desplaza el
    # rango del dataset de [2024-11-26, 2026-04-26] a [2024-01-12, 2026-12-02],
    # produciendo 21 "fechas futuras" que no existen. En el dataset limpio no se
    # nota porque ya esta en ISO, pero en la Fase 4 esta funcion validara
    # clientes nuevos con el formato de origen.
    'fecha_prestamo':        {'min_fecha': '2020-01-01', 'max_fecha': 'hoy', 'nulos_permitidos': False,
                              'dayfirst': True,
                              'fuente': 'Una fecha de desembolso no puede ser futura ni anterior al '
                                        'inicio de operacion del producto. Rango observado: '
                                        '2024-11-26 a 2026-04-26'},
}


def _a_fecha(serie: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Convierte a datetime sin depender de la inferencia de formato de pandas.

    La misma columna llega en dos formatos segun la etapa del pipeline: el
    archivo de origen la trae como d/m/Y ("7/01/2025 14:40"), y el dataset ya
    procesado la trae en ISO ("2025-01-07 14:40:00"). Dejar que pandas adivine
    falla en ambos sentidos: sin `dayfirst` lee el 7 de enero como 1 de julio;
    con `dayfirst` sobre ISO intenta %Y-%d-%m y desordena mes y dia.

    Se resuelve en dos pasadas deterministas: ISO primero, y solo lo que no
    encaje se reintenta con la convencion declarada en la regla.
    """
    fechas = pd.to_datetime(serie, errors='coerce', format='ISO8601')
    pendientes = fechas.isna() & serie.notna()
    if pendientes.any():
        fechas = fechas.copy()
        fechas[pendientes] = pd.to_datetime(serie[pendientes], errors='coerce',
                                            dayfirst=dayfirst)
    return fechas


def validar_dataframe(datos: pd.DataFrame, reglas: dict | None = None) -> list[str]:
    """Valida un DataFrame contra el contrato. Devuelve la lista de violaciones.

    Una lista vacia significa que los datos cumplen todas las reglas definidas.
    Las columnas que no aparecen en `reglas` no se validan; las que aparecen en
    `reglas` pero faltan en el DataFrame se reportan como ausentes.
    """
    reglas = REGLAS_VALIDACION if reglas is None else reglas
    violaciones: list[str] = []

    for col, regla in reglas.items():
        if col not in datos.columns:
            violaciones.append(f"{col}: columna ausente")
            continue

        if not regla.get('nulos_permitidos', True) and datos[col].isnull().any():
            violaciones.append(f"{col}: {datos[col].isnull().sum()} nulos no permitidos")

        serie = datos[col].dropna()

        if 'min_fecha' in regla:
            # Regla temporal: la fecha no puede ser futura ni anterior al inicio
            # de operacion del producto.
            limite_inf = pd.Timestamp(regla['min_fecha'])
            limite_sup = (pd.Timestamp.today() if regla['max_fecha'] == 'hoy'
                          else pd.Timestamp(regla['max_fecha']))
            serie_fecha = _a_fecha(serie, dayfirst=regla.get('dayfirst', False))
            if (serie_fecha < limite_inf).any():
                violaciones.append(f"{col}: {(serie_fecha < limite_inf).sum()} "
                                   f"fechas anteriores a {regla['min_fecha']}")
            if (serie_fecha > limite_sup).any():
                violaciones.append(f"{col}: {(serie_fecha > limite_sup).sum()} fechas futuras")

        elif 'valores' in regla:
            invalidos = ~serie.isin(regla['valores'])
            if invalidos.any():
                violaciones.append(f"{col}: {invalidos.sum()} valores fuera del dominio permitido")

        else:
            if regla.get('min') is not None and (serie < regla['min']).any():
                violaciones.append(f"{col}: {(serie < regla['min']).sum()} "
                                   f"valores bajo el minimo ({regla['min']})")
            if regla.get('max') is not None and (serie > regla['max']).any():
                violaciones.append(f"{col}: {(serie > regla['max']).sum()} "
                                   f"valores sobre el maximo ({regla['max']})")

    return violaciones
