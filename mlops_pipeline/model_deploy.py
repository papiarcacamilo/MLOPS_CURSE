"""
Despliegue del modelo
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/models/modelo_seleccionado.joblib + reglas_negocio.py
Salida   : motor de inferencia por lote, artefacto de despliegue y registro
Estado   : IMPLEMENTADA

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Se toma el mejor modelo desplegado y una imagen que contenga las librerias
     y el codigo para una app que permita disponibilizar dicho objeto y despliega
     el modelo en un endpoint que puede utilizarse para predicciones (por batch)."

Son tres piezas, y cada una vive en su archivo:

    la app        app.py       API con el endpoint por lote
    la imagen     Dockerfile   librerias + codigo + artefactos
    el motor      este archivo la cadena de inferencia que la app expone

EL REQUISITO QUE NO PUEDE FALLAR

Un cliente nuevo debe atravesar EXACTAMENTE las mismas transformaciones que el
conjunto de entrenamiento. Si el endpoint recalcula cortes, reajusta WoE o
imputa de otra forma, el modelo recibe una entrada que no se parece a lo que
aprendio (train-serving skew) y falla en silencio: sigue devolviendo
probabilidades, solo que equivocadas.

Por eso aqui NADA se ajusta. `cadena_inferencia()` reutiliza los pasos ya
ajustados del joblib y les antepone las derivadas, que son fila a fila. Y las
constantes de saneamiento se congelan en el artefacto en lugar de recalcularse
sobre el lote que llega: una mediana estimada sobre 50 solicitudes no es la
mediana con la que el modelo aprendio.

ORDEN DE LA CADENA

    1. Sanear      aplicar las correcciones que la Fase 1 definio, con las
                   constantes congeladas: edad imposible, salario cero o
                   extremo, tendencia corrupta, score fuera del rango oficial
    2. Validar     contra REGLAS_VALIDACION del contrato, fila a fila. Lo que
                   incumple se rechaza, no se puntua
    3. Transformar aplicar el ColumnTransformer ajustado: derivadas, binning,
                   WoE, escalado
    4. Puntuar     probabilidad de mora y su traduccion a puntos de scorecard
    5. Decidir     umbral de operacion 0.0661, fijado en la evaluacion
    6. Registrar   guardar entrada y prediccion: es el insumo de
                   model_monitoring.py, que sin esta tabla no puede medir nada

POR QUE SANEAR VA ANTES DE VALIDAR

El docstring original tenia el orden inverso, y al implementarlo se vio que no
funciona. El contrato exige `puntaje_datacredito` entre 150 y 950, pero la Fase 1
documento que un valor fuera de ese rango no es un error: es ausencia de
historial, y su tratamiento definido es pasar a nulo con bandera. Validar
primero rechazaria como invalido justo lo que el EDA decidio conservar, y el
modelo se quedaria sin el 1.4% de solicitudes que mas informacion aportan.

La distincion es entre dos clases de anomalia:

    con tratamiento definido   edad > 90, salario 0 o extremo, tendencia
                               corrupta, score fuera de rango. El EDA decidio
                               que hacer y el modelo se entreno sobre el
                               resultado. El endpoint repite ese tratamiento
    sin tratamiento definido   columna ausente, edad de 12 anios, plazo 0,
                               tipo_credito 99. No hay regla que aplicar, y
                               puntuar seria inventar. Rechazo tecnico

`verificar_saneamiento()` es el criterio de aceptacion: sanear los 10.763
registros crudos debe reproducir Base_de_datos_limpia.csv columna por columna.
Si difiere en un solo valor, el endpoint esta sirviendo otra cosa.

SCORECARD (etapa 10)

La tabla de puntos se calcula en model_evaluation.py, junto con la verificacion
de que reproduce la prediccion del modelo. Aqui se PUBLICA como artefacto de
despliegue y se aplica solicitud a solicitud: cada respuesta trae el puntaje y
las tres variables que mas lo mueven. Es el formato con el que un analista de
riesgo justifica una negacion ante el cliente y ante el supervisor.

En credito la interpretabilidad no es un empate tecnico, es una ventaja: ante
una ganancia marginal de AUC-PR, gana el modelo simple.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    _DIR_MODULO = str(Path(__file__).resolve().parent)
except NameError:
    _DIR_MODULO = str(Path.cwd().resolve())
if _DIR_MODULO not in sys.path:
    sys.path.insert(0, _DIR_MODULO)

import ft_engineering as fe  # noqa: E402
import reglas_negocio as rn  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("despliegue")

RUTA_RAIZ = fe.RUTA_RAIZ
RUTA_MODELOS = fe.RUTA_MODELOS
RUTA_DATOS = fe.RUTA_SALIDA
RUTA_MONITOREO = RUTA_RAIZ / "data" / "monitoring"

RUTA_MODELO = RUTA_MODELOS / "modelo_seleccionado.joblib"
RUTA_EVALUACION = RUTA_MODELOS / "evaluacion.json"
RUTA_ARTEFACTO = RUTA_MODELOS / "artefacto_despliegue.json"
RUTA_SCORECARD = RUTA_MODELOS / "scorecard.csv"
RUTA_REGISTRO = RUTA_MONITOREO / "registro_endpoint.csv"
RUTA_LOTE_EJEMPLO = RUTA_MONITOREO / "lote_ejemplo.csv"

TARGET = fe.TARGET
COLUMNA_FECHA = fe.COLUMNA_FECHA


# ==============================================================================
# CONSTANTES CONGELADAS
# ==============================================================================
#
# Cada una se estimo UNA vez, sobre el dataset completo, en la Fase 1. Aqui se
# transcriben con su procedencia para que el saneamiento en produccion sea
# identico al de entrenamiento. Recalcularlas sobre el lote entrante seria la
# forma mas facil de introducir train-serving skew sin que nada falle.

SANEAMIENTO = {
    # comprension_eda.ipynb, celda 28. No hay registros entre 70 y 120 anios,
    # de modo que el umbral separa los atipicos (121, 122, 123) sin tocar
    # edades reales. Los 150 marcados se imputan con la mediana de los validos.
    "edad_umbral": 90,
    "edad_mediana": 42.0,

    # celda 32. IQR sobre escala logaritmica, porque el IQR directo sobremarca
    # salarios altos legitimos: la distribucion es asimetrica por naturaleza.
    # Marca tambien el salario cero, que no es un ingreso sino un dato ausente.
    "salario_limite_superior": 18790797.2489,
    "salario_mediana": 3000000.0,

    # celda 12 y celdas 21-22, en ese orden. Primero se rellenan los 6 nulos
    # originales con la mediana; despues el rango oficial de DataCredito
    # Experian convierte en nulo lo que queda fuera, con bandera explicita.
    # Invertir estos dos pasos cambia 6 registros.
    "puntaje_mediana": 791.0,
    "score_min": 150,
    "score_max": 950,

    # celda 35. Dominio cerrado; lo que no pertenece a el se reconstruye por el
    # signo del valor numerico que traia.
    "tendencia_validas": ["Creciente", "Decreciente", "Estable", "Sin_dato"],

    # celda 82. Se marca sin imputar: no hay regla de negocio que distinga un
    # endeudamiento corporativo legitimo de un error de digitacion.
    "otros_prestamos_limite": 1_000_000_000,
    "ratio_deuda_salario_max": 100,
}

# Ultima fecha observada en el dataset completo. Solo interviene en
# `antiguedad_dias`, que NO es una de las 17 caracteristicas del modelo, pero se
# congela igual: sin ella `construir_derivadas` no se puede invocar.
FECHA_CORTE_OBS = pd.Timestamp("2026-04-26")

# Lo que una solicitud debe traer. Es un subconjunto del archivo de origen:
# faltan las de fuga (saldos posteriores al desembolso), las descartadas por
# medicion en la Fase 2 y la variable objetivo, que en originacion no existe.
COLUMNAS_REQUERIDAS = [
    "tipo_credito", "fecha_prestamo", "capital_prestado", "plazo_meses",
    "edad_cliente", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta",
    "promedio_ingresos_datacredito", "tendencia_ingresos",
]

# Las dos que admiten nulo, por decision del EDA: su ausencia es informacion y
# el WoE la trata como un tramo mas.
NULOS_ADMITIDOS = ["puntaje_datacredito", "promedio_ingresos_datacredito"]

# Reglas del contrato que aplican a una solicitud de originacion. Se filtran las
# del objetivo y las de saldos, que no llegan al endpoint.
REGLAS_ENDPOINT = {c: r for c, r in rn.REGLAS_VALIDACION.items()
                   if c in COLUMNAS_REQUERIDAS}

# UNICA excepcion, y hay que justificarla. El contrato pone techo de 1.000
# millones a `total_otros_prestamos` para SENIALAR los 13 registros no
# verificables, no para excluirlos: la Fase 1 decidio marcarlos sin imputar,
# precisamente porque no hay regla que distinga deuda corporativa legitima de un
# error de digitacion. Esos 13 registros entraron al entrenamiento con su
# bandera puesta.
#
# Aplicar el techo como rechazo tecnico rompia esa decision por partida doble:
# negaba en produccion lo que en entrenamiento se puntuo, y dejaba
# `total_otros_prestamos_sospechoso` sin poder activarse nunca por esa via. El
# minimo si se conserva: una deuda negativa no tiene tratamiento definido.
REGLAS_ENDPOINT["total_otros_prestamos"] = {
    **rn.REGLAS_VALIDACION["total_otros_prestamos"],
    "max": None,
    "fuente": ("Un monto de deuda no puede ser negativo. El techo de 1.000M del "
               "contrato se traslada a la bandera "
               "total_otros_prestamos_sospechoso, que es el tratamiento que "
               "definio la Fase 1."),
}

DECISION_APROBAR = "APROBAR"
DECISION_RECHAZAR = "RECHAZAR"
DECISION_TECNICO = "RECHAZO_TECNICO"
MOTIVO_POLITICA = "politica: sin score de central de riesgo"


# Los artefactos se leen una vez por proceso. `limpiar_cache()`, definido junto a
# las funciones que los cargan, es la unica forma de releerlos.


# ==============================================================================
# 1 | SANEAR
# ==============================================================================


def sanear(datos: pd.DataFrame) -> pd.DataFrame:
    """Aplica las correcciones de la Fase 1 con las constantes congeladas.

    El orden importa y es el del EDA: la bandera de `total_otros_prestamos`
    compara contra el salario YA corregido, y el nulo del score se decide
    despues de rellenar los nulos originales con la mediana.

    Devuelve el DataFrame con las columnas corregidas y las cinco banderas de
    trazabilidad, cuatro de las cuales son caracteristicas del modelo.

    ES IDEMPOTENTE, y no lo era. Sanear un lote YA saneado apagaba todas las
    banderas, porque se recalculan comparando contra los umbrales y sobre valores
    ya corregidos ninguna se dispara, y ademas rellenaba con la mediana los nulos
    que marcaban ausencia de historial. Medido sobre el train: 103 banderas de
    edad, 187 de salario, 47 de tendencia y 105 nulos del score, todos perdidos
    en la segunda pasada, con probabilidades distintas y sin un solo error.

    En originacion el endpoint recibe datos crudos y el caso no aparece. Pero un
    reenvio, una integracion que limpie aguas arriba o el propio monitoreo
    reproduciendo la cadena bastan para dispararlo, y el sintoma seria un
    puntaje que cambia sin que nada haya cambiado. Por eso cada bandera que el
    lote ya traiga se conserva en vez de recalcularse a ciegas.
    """
    d = datos.copy()
    c = SANEAMIENTO

    def _bandera_previa(nombre: str) -> pd.Series:
        """Bandera que el lote ya traia, o todo apagado si no venia."""
        if nombre not in datos.columns:
            return pd.Series(False, index=d.index)
        return datos[nombre].astype(str).str.lower().isin(["true", "1"])

    # La fecha llega en el formato de origen (d/m/Y). El contrato declara la
    # convencion en vez de dejar que pandas la adivine.
    if COLUMNA_FECHA in d.columns:
        d[COLUMNA_FECHA] = rn._a_fecha(d[COLUMNA_FECHA], dayfirst=True)

    # Edad
    d["edad_cliente"] = pd.to_numeric(d["edad_cliente"], errors="coerce")
    d["edad_cliente_corregida"] = (
        (d["edad_cliente"] > c["edad_umbral"]) | _bandera_previa("edad_cliente_corregida"))
    d.loc[d["edad_cliente"] > c["edad_umbral"], "edad_cliente"] = c["edad_mediana"]

    # Salario
    d["salario_cliente"] = pd.to_numeric(d["salario_cliente"], errors="coerce")
    fuera_salario = ((d["salario_cliente"] == 0)
                     | (d["salario_cliente"] > c["salario_limite_superior"]))
    d["salario_cliente_corregido"] = (
        fuera_salario | _bandera_previa("salario_cliente_corregido"))
    d.loc[fuera_salario, "salario_cliente"] = c["salario_mediana"]

    # Tendencia de ingresos
    tendencia = d["tendencia_ingresos"]
    corrupto = tendencia.notna() & ~tendencia.isin(c["tendencia_validas"])
    d["tendencia_ingresos_reconstruida"] = (
        corrupto | _bandera_previa("tendencia_ingresos_reconstruida"))
    numerico = pd.to_numeric(tendencia[corrupto], errors="coerce")
    d.loc[corrupto, "tendencia_ingresos"] = np.select(
        [numerico > 0, numerico < 0], ["Creciente", "Decreciente"], "Estable")
    d["tendencia_ingresos"] = d["tendencia_ingresos"].fillna("Sin_dato")

    # Score de central de riesgo. El nulo que ya venia marcado como ausencia de
    # historial se respeta: rellenarlo con la mediana convertiria un "no tiene
    # score" en un score promedio inventado.
    puntaje = pd.to_numeric(d["puntaje_datacredito"], errors="coerce")
    sin_historial_previo = _bandera_previa("sin_historial_crediticio")
    puntaje = puntaje.where(sin_historial_previo, puntaje.fillna(c["puntaje_mediana"]))
    fuera = (puntaje < c["score_min"]) | (puntaje > c["score_max"])
    d["sin_historial_crediticio"] = (fuera | sin_historial_previo).astype(int)
    d["puntaje_datacredito"] = puntaje.mask(fuera)

    # Ingreso reportado por el buro: no se imputa, se marca
    d["promedio_ingresos_datacredito"] = pd.to_numeric(
        d["promedio_ingresos_datacredito"], errors="coerce")
    d["promedio_ingresos_datacredito_era_nulo"] = \
        d["promedio_ingresos_datacredito"].isnull()

    # Otras obligaciones
    d["total_otros_prestamos"] = pd.to_numeric(
        d["total_otros_prestamos"], errors="coerce")
    ratio = d["total_otros_prestamos"] / d["salario_cliente"]
    d["total_otros_prestamos_sospechoso"] = (
        (d["total_otros_prestamos"] > c["otros_prestamos_limite"])
        | (ratio > c["ratio_deuda_salario_max"])
    ).astype(int)

    return d


def verificar_saneamiento() -> dict:
    """Criterio de aceptacion: sanear el crudo debe reproducir el limpio.

    Es la unica prueba que garantiza que el endpoint aplica el mismo
    preprocesamiento que produjo los datos de entrenamiento. Se ejecuta sobre
    los 10.763 registros, no sobre una muestra.

    Comprueba ademas la IDEMPOTENCIA: sanear dos veces debe dar lo mismo que
    sanear una. Sin esa segunda comprobacion, la funcion podia reproducir el
    limpio a la perfeccion y aun asi destruirlo al recibirlo de vuelta.
    """
    crudo = pd.read_csv(RUTA_RAIZ / fe.CONFIG["data"]["raw_path"],
                        sep=fe.SEPARADOR, encoding=fe.ENCODING)
    limpio = pd.read_csv(RUTA_RAIZ / fe.CONFIG["data"]["clean_path"],
                         sep=fe.SEPARADOR, encoding=fe.ENCODING)
    saneado = sanear(crudo)

    comparables = [
        "edad_cliente", "salario_cliente", "tendencia_ingresos",
        "puntaje_datacredito", "edad_cliente_corregida",
        "salario_cliente_corregido", "tendencia_ingresos_reconstruida",
        "sin_historial_crediticio", "total_otros_prestamos_sospechoso",
        "promedio_ingresos_datacredito_era_nulo",
    ]

    def comparar(izquierda: pd.DataFrame, derecha: pd.DataFrame) -> dict:
        salida = {}
        for col in comparables:
            a, b = izquierda[col], derecha[col]
            if a.dtype == bool or b.dtype == bool:
                a, b = a.astype(float), b.astype(float)
            salida[col] = int((~((a == b) | (a.isna() & b.isna()))).sum())
        return salida

    diferencias = comparar(saneado, limpio)

    # Segunda pasada sobre el resultado de la primera. Debe devolver lo mismo:
    # un endpoint cuyo saneamiento no sea idempotente cambia el puntaje de una
    # solicitud reenviada sin que nada haya cambiado en el cliente.
    diferencias_segunda = comparar(sanear(saneado), saneado)

    return {
        "registros": int(len(crudo)),
        "columnas_comparadas": len(comparables),
        "diferencias": diferencias,
        "reproduce_el_limpio": all(v == 0 for v in diferencias.values()),
        "diferencias_segunda_pasada": diferencias_segunda,
        "idempotente": all(v == 0 for v in diferencias_segunda.values()),
    }


# ==============================================================================
# 2 | VALIDAR
# ==============================================================================


def _violacion_de_fila(valor, columna: str, regla: dict) -> str:
    """Evalua una regla del contrato sobre un valor. Cadena vacia si cumple."""
    if pd.isna(valor):
        if regla.get("nulos_permitidos", True):
            return ""
        return f"{columna}: nulo no permitido"

    if "valores" in regla:
        if valor not in regla["valores"]:
            return f"{columna}: valor fuera del dominio permitido"
        return ""

    if "min_fecha" in regla:
        limite_inf = pd.Timestamp(regla["min_fecha"])
        limite_sup = (pd.Timestamp.today() if regla["max_fecha"] == "hoy"
                      else pd.Timestamp(regla["max_fecha"]))
        fecha = pd.Timestamp(valor)
        if fecha < limite_inf:
            return f"{columna}: anterior a {regla['min_fecha']}"
        if fecha > limite_sup:
            return f"{columna}: fecha futura"
        return ""

    if regla.get("min") is not None and valor < regla["min"]:
        return f"{columna}: bajo el minimo ({regla['min']})"
    if regla.get("max") is not None and valor > regla["max"]:
        return f"{columna}: sobre el maximo ({regla['max']})"
    return ""


def verificar_esquema(datos: pd.DataFrame) -> list[str]:
    """Columnas requeridas que el lote no trae.

    Se comprueba ANTES de sanear, porque `sanear` da por hecho que las columnas
    existen: un archivo con otro esquema reventaba ahi con un KeyError, que el
    endpoint traducia a un error 500. Un esquema equivocado es culpa de quien
    llama, no del servicio, y debe decirlo con un 400 y el nombre de lo que
    falta.

    Es distinto de `validar_filas`: alli falla una solicitud y las demas siguen;
    aqui no hay lote que procesar.
    """
    return [c for c in COLUMNAS_REQUERIDAS if c not in datos.columns]


def validar_filas(datos: pd.DataFrame) -> pd.Series:
    """Valida fila a fila contra el contrato. Devuelve el motivo, o vacio.

    `validar_dataframe()` del contrato responde a nivel de lote, que es lo que
    necesita una compuerta de calidad. Un endpoint necesita granularidad de
    fila: una solicitud invalida no puede tumbar las otras 49 del lote.

    Las reglas son las mismas; lo que cambia es el alcance de la respuesta.
    """
    faltantes = verificar_esquema(datos)
    if faltantes:
        return pd.Series(f"columnas ausentes: {', '.join(faltantes)}",
                         index=datos.index)

    motivos = pd.Series("", index=datos.index, dtype=object)
    for columna, regla in REGLAS_ENDPOINT.items():
        fallos = datos[columna].map(
            lambda v, c=columna, r=regla: _violacion_de_fila(v, c, r))
        vacios = motivos == ""
        motivos = motivos.where(~(vacios & (fallos != "")), fallos)
    return motivos


def calidad_del_lote(datos: pd.DataFrame) -> list[str]:
    """Violaciones del lote ANTES de sanear, a nivel de columna.

    No decide nada: un registro con score fuera de rango se sanea y se puntua
    igual. Se mide porque es una senial de calidad de la fuente, y porque
    model_monitoring.py la necesita para distinguir un cambio real de poblacion
    de un cambio en como llega el archivo.
    """
    return rn.validar_dataframe(datos, REGLAS_ENDPOINT)


# ==============================================================================
# 3 | TRANSFORMAR Y PUNTUAR
# ==============================================================================


@lru_cache(maxsize=1)
def version_modelo() -> str:
    """Hash corto del joblib. Identifica que objeto respondio cada solicitud."""
    return hashlib.sha256(RUTA_MODELO.read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=1)
def cadena_inferencia() -> Pipeline:
    """Pipeline de servicio: derivadas + caracteristicas ajustadas + modelo.

    Los dos ultimos pasos se toman TAL CUAL del joblib, ya ajustados. El primero
    no aprende nada: `construir_derivadas` es fila a fila.

    Construirla asi, y no llamando a `fe.construir_pipeline(con_derivadas=True)`,
    es deliberado: esa funcion devuelve un pipeline SIN ajustar, y usarlo exigiria
    un fit que aqui no debe existir.

    Cacheado: sin `lru_cache` cada peticion deserializaba el joblib de nuevo, y
    medido sobre lotes de 50 solicitudes eso era la mayor parte de la latencia.
    El objeto es inmutable, de modo que cachearlo no cambia ninguna respuesta.
    Cambiar el modelo exige reiniciar el proceso, que es la semantica correcta
    para un modelo servido: nadie quiere que el artefacto cambie a mitad de un
    lote.
    """
    modelo = joblib.load(RUTA_MODELO)
    return Pipeline([
        ("derivadas", fe.ConstructorDerivadas(FECHA_CORTE_OBS)),
        ("caracteristicas", modelo.named_steps["caracteristicas"]),
        ("modelo", modelo.named_steps["modelo"]),
    ])


# ==============================================================================
# SCORECARD APLICADO
# ==============================================================================


@lru_cache(maxsize=1)
def cargar_scorecard() -> dict:
    """Lee la tabla de puntos que produjo y verifico model_evaluation.py."""
    with open(RUTA_EVALUACION, encoding="utf-8") as f:
        return json.load(f)["scorecard"]


def puntaje_desde_probabilidad(p: np.ndarray, escala: dict) -> np.ndarray:
    """Traduce probabilidad a puntos, con la escala del sector.

        puntaje = offset - factor * ln(odds de mora)

    Mas puntos significan menos riesgo. Es la misma transformacion que
    `verificar_scorecard` comprobo contra la suma de la tabla, con error maximo
    de 2.2e-16 en probabilidad.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    z = np.log(p / (1 - p))
    return escala["offset"] - escala["factor"] * z


def razones(matriz: pd.DataFrame, scorecard: dict, n: int = 3) -> pd.Series:
    """Las n variables que mas mueven el puntaje de cada solicitud.

    Es la traduccion del modelo a lenguaje de comite de credito: en lugar de
    "probabilidad 0.09", la respuesta dice que plazo y score restaron puntos y
    la edad los sumo. Sin esto el scorecard seria decorativo.

    El signo se lee desde el puntaje, no desde el WoE: un aporte negativo resta
    puntos, es decir, empeora el perfil.
    """
    factor = scorecard["escala"]["factor"]
    coeficientes = {f"woe_{fila['variable']}": fila["coeficiente"]
                    for fila in scorecard["tabla"]}
    coeficientes.update({fila["variable"]: fila["coeficiente"]
                         for fila in scorecard["ajustes_continuos"]})

    presentes = [c for c in matriz.columns if c in coeficientes]
    aportes = matriz[presentes].astype(float) * [-factor * coeficientes[c]
                                                 for c in presentes]
    etiquetas = [c[4:] if c.startswith("woe_") else c for c in presentes]
    aportes.columns = etiquetas

    def describir(fila: pd.Series) -> str:
        top = fila.reindex(fila.abs().sort_values(ascending=False).index)[:n]
        return "; ".join(f"{k} {v:+.0f}" for k, v in top.items())

    return aportes.apply(describir, axis=1)


def bandas_desde_train(puntajes: np.ndarray, objetivo: np.ndarray,
                       n_bandas: int = 5) -> list[dict]:
    """Bandas de riesgo por quintil del puntaje de entrenamiento.

    Los cortes se miden, no se inventan: son los quintiles observados, y cada
    banda se publica con la tasa de mora que le corresponde en train. Asi la
    letra que ve el analista tiene un numero detras.
    """
    bordes = np.quantile(puntajes, np.linspace(0, 1, n_bandas + 1))
    bordes[0], bordes[-1] = -np.inf, np.inf
    nombres = ["E", "D", "C", "B", "A"]  # menos puntos, mas riesgo

    tramos = pd.cut(puntajes, bordes, labels=nombres, include_lowest=True)
    tabla = pd.DataFrame({"banda": tramos, "mora": objetivo})
    resumen = tabla.groupby("banda", observed=False)["mora"].agg(["size", "mean"])

    return [{
        "banda": nombre,
        "puntaje_desde": None if i == 0 else round(float(bordes[i]), 1),
        "puntaje_hasta": None if i == n_bandas - 1 else round(float(bordes[i + 1]), 1),
        "n_train": int(resumen.loc[nombre, "size"]),
        "tasa_mora_train": round(float(resumen.loc[nombre, "mean"]), 4),
    } for i, nombre in enumerate(nombres)]


def asignar_banda(puntajes: np.ndarray, bandas: list[dict]) -> np.ndarray:
    """Ubica cada puntaje en su banda."""
    bordes = [-np.inf] + [b["puntaje_hasta"] for b in bandas[:-1]] + [np.inf]
    nombres = [b["banda"] for b in bandas]
    return pd.cut(puntajes, bordes, labels=nombres,
                  include_lowest=True).astype(str)


# ==============================================================================
# 4 | DECIDIR
# ==============================================================================


@lru_cache(maxsize=1)
def cargar_artefacto() -> dict:
    """Lee el artefacto de despliegue. Falla claro si no se ha construido."""
    if not RUTA_ARTEFACTO.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_ARTEFACTO.name}. Ejecuta primero:\n"
            f"    python mlops_pipeline/model_deploy.py")
    with open(RUTA_ARTEFACTO, encoding="utf-8") as f:
        return json.load(f)


def limpiar_cache() -> None:
    """Olvida los artefactos cacheados y obliga a releerlos del disco.

    Solo hace falta cuando el mismo proceso REESCRIBE un artefacto y despues lo
    consume, que es justo lo que hace `main()`. El endpoint no la llama: ahi la
    inmutabilidad durante la vida del proceso es lo que se quiere.
    """
    for f in (version_modelo, cadena_inferencia, cargar_scorecard, cargar_artefacto):
        f.cache_clear()


def decidir(probabilidad: np.ndarray, umbral: float, sin_score: np.ndarray,
            motivo_tecnico: pd.Series,
            rechazar_sin_score: bool = rn.RECHAZAR_SIN_SCORE
            ) -> tuple[np.ndarray, np.ndarray]:
    """Aplica el umbral de operacion y la politica del contrato.

    Devuelve la decision y la mascara de solicitudes que rechazo la POLITICA
    pese a quedar por debajo del umbral. Esa mascara no es un detalle interno:
    sin ella la respuesta mostraria una probabilidad baja junto a un rechazo,
    sin nada que explique la contradiccion.

    El umbral 0.0661 no se elige aqui: viene de model_evaluation.py, fijado
    sobre predicciones fuera de fold para rechazar el mismo 20.9% que rechazaba
    la regla heuristica. Elegirlo en despliegue, con los datos que llegan,
    seria ajustar el punto de operacion a la poblacion que se quiere medir.

    `RECHAZAR_SIN_SCORE` es politica de negocio publicada en el contrato, no una
    salida del modelo. El modelo ya penaliza la ausencia de score (WoE +0.417),
    de modo que la politica solo agrega el 0.40% de la cartera que aun quedaba
    por debajo del umbral. Se aplica despues de puntuar y la probabilidad se
    reporta igual, para que la decision sea auditable y no una caja cerrada.
    """
    decision = np.where(probabilidad >= umbral, DECISION_RECHAZAR, DECISION_APROBAR)

    sin_score = sin_score.astype(bool)
    por_politica = np.zeros(len(decision), dtype=bool)
    if rechazar_sin_score:
        por_politica = sin_score & (probabilidad < umbral)
        decision = np.where(sin_score, DECISION_RECHAZAR, decision)

    tecnico = motivo_tecnico.to_numpy() != ""
    return (np.where(tecnico, DECISION_TECNICO, decision),
            por_politica & ~tecnico)


# ==============================================================================
# 5 | REGISTRAR
# ==============================================================================


def registrar(entrada: pd.DataFrame, salida: pd.DataFrame, origen: str) -> Path:
    """Anexa entradas y pronosticos al registro del endpoint.

    Es literalmente lo que pide la Fase 5: "una tabla los datos pasados al
    endpoint junto con los pronosticos entregados por este". Se guarda la
    entrada TAL COMO LLEGO, antes de sanear, porque el saneamiento borra la
    evidencia de como venia la fuente, y esa evidencia es una de las seniales
    de deriva.
    """
    RUTA_MONITOREO.mkdir(parents=True, exist_ok=True)
    columnas_entrada = [c for c in COLUMNAS_REQUERIDAS if c in entrada.columns]

    fila = entrada[columnas_entrada].reset_index(drop=True)
    fila.insert(0, "momento", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    fila.insert(1, "origen", origen)
    fila.insert(2, "version_modelo", version_modelo())
    registro = pd.concat([fila, salida.reset_index(drop=True)], axis=1)

    registro.to_csv(RUTA_REGISTRO, sep=fe.SEPARADOR, encoding=fe.ENCODING,
                    index=False, mode="a", header=not RUTA_REGISTRO.exists())
    return RUTA_REGISTRO


# ==============================================================================
# ORQUESTACION DE LA INFERENCIA
# ==============================================================================


def predecir_lote(solicitudes: pd.DataFrame, origen: str = "batch",
                  guardar_registro: bool = True) -> pd.DataFrame:
    """Cadena completa sobre un lote de solicitudes crudas.

    Devuelve una fila por solicitud, en el mismo orden que entro. Ninguna
    columna queda vacia: una solicitud rechazada por contrato trae el motivo, y
    una puntuada trae sus tres factores.
    """
    if solicitudes.empty:
        raise ValueError("El lote no contiene solicitudes.")

    faltantes = verificar_esquema(solicitudes)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")

    artefacto = cargar_artefacto()
    scorecard = cargar_scorecard()
    escala = scorecard["escala"]

    saneado = sanear(solicitudes)
    motivo_tecnico = validar_filas(saneado)
    valido = motivo_tecnico == ""

    n = len(saneado)
    probabilidad = np.full(n, np.nan)
    puntaje = np.full(n, np.nan)
    motivo = motivo_tecnico.to_numpy().astype(object)

    if valido.any():
        cadena = cadena_inferencia()
        aptos = saneado.loc[valido]
        probabilidad[valido.to_numpy()] = cadena.predict_proba(aptos)[:, 1]
        puntaje[valido.to_numpy()] = puntaje_desde_probabilidad(
            probabilidad[valido.to_numpy()], escala)

        matriz = Pipeline(cadena.steps[:-1]).transform(aptos)
        motivo[valido.to_numpy()] = razones(matriz, scorecard).to_numpy()

    decision, por_politica = decidir(
        np.nan_to_num(probabilidad, nan=1.0),
        artefacto["umbral"]["valor"],
        saneado["sin_historial_crediticio"].to_numpy(),
        motivo_tecnico)

    # Un rechazo por politica con probabilidad baja parece un error si no se
    # explica. El motivo lo declara antes que los factores del scorecard.
    motivo[por_politica] = [f"{MOTIVO_POLITICA}; {m}" for m in motivo[por_politica]]

    banda = np.where(valido.to_numpy(),
                     asignar_banda(puntaje, artefacto["bandas"]),
                     "SIN_BANDA")

    salida = pd.DataFrame({
        "decision": decision,
        "probabilidad_mora": np.round(probabilidad, 6),
        "puntaje": np.round(puntaje, 1),
        "banda": banda,
        "motivo": motivo,
    })

    if guardar_registro:
        registrar(solicitudes, salida, origen)
    return salida


# ==============================================================================
# CONSTRUCCION DEL ARTEFACTO
# ==============================================================================


def construir_artefacto() -> dict:
    """Congela en un JSON todo lo que el endpoint necesita para responder.

    El joblib guarda el modelo; este archivo guarda el resto del contrato de
    servicio: que columnas se exigen, con que constantes se sanea, en que
    umbral se decide y sobre que bandas se reporta. Sin el, la app tendria que
    recalcular decisiones que ya se tomaron en fases anteriores.
    """
    with open(RUTA_EVALUACION, encoding="utf-8") as f:
        evaluacion = json.load(f)

    modelo = joblib.load(RUTA_MODELO)
    train = pd.read_csv(RUTA_DATOS / "estratificado_train.csv",
                        sep=fe.SEPARADOR, encoding=fe.ENCODING)
    X, y = train.drop(columns=[TARGET]), (1 - train[TARGET]).to_numpy()

    p_train = modelo.predict_proba(X)[:, 1]
    puntajes = puntaje_desde_probabilidad(p_train, evaluacion["scorecard"]["escala"])

    interno = modelo.named_steps["modelo"].named_steps["logisticregression"]
    metricas = evaluacion["evaluacion_final"]["estratificado"]["metricas"]

    return {
        "modelo": {
            "nombre": evaluacion["modelo"],
            "archivo": RUTA_MODELO.name,
            "version": version_modelo(),
            "ajustado_sobre": "estratificado_train.csv (8.610 registros)",
            "n_caracteristicas": int(interno.coef_.shape[1]),
            "nota": ("Es el mismo objeto que se evaluo sobre test. No se "
                     "reajusta con train+test: reajustarlo mejoraria el modelo "
                     "servido pero romperia la trazabilidad entre lo que se "
                     "midio y lo que responde el endpoint."),
        },
        "desempenio_en_test": {k: metricas[k] for k in
                               ("auc_pr", "ks", "gini", "brier", "recall",
                                "precision", "pct_marcado")},
        "umbral": {
            "valor": evaluacion["umbral"]["umbral"],
            "criterio": evaluacion["umbral"]["criterio"],
            "fijado_en": "model_evaluation.py, sobre predicciones fuera de fold",
        },
        "politica": {
            "rechazar_sin_score": rn.RECHAZAR_SIN_SCORE,
            "fuente": "reglas_negocio.py, contrato del EDA",
        },
        "esquema_entrada": {
            "columnas_requeridas": COLUMNAS_REQUERIDAS,
            "nulos_admitidos": NULOS_ADMITIDOS,
            "separador": fe.SEPARADOR,
            "encoding": fe.ENCODING,
            "formato_fecha": "d/m/Y, declarado en el contrato",
        },
        "saneamiento": SANEAMIENTO,
        "fecha_corte_observacion": str(FECHA_CORTE_OBS.date()),
        "bandas": bandas_desde_train(puntajes, y),
        "escala_scorecard": evaluacion["scorecard"]["escala"],
    }


# ==============================================================================
# ORQUESTACION
# ==============================================================================


def main() -> dict:
    log.info("=" * 72)
    log.info("FASE 4 | DESPLIEGUE")
    log.info("=" * 72)

    # --- Criterio de aceptacion ---------------------------------------------
    log.info("PASO 1 | El saneamiento reproduce la limpieza de la Fase 1")
    verificacion = verificar_saneamiento()
    if verificacion["reproduce_el_limpio"]:
        log.info("   %s registros, %s columnas, 0 diferencias",
                 verificacion["registros"], verificacion["columnas_comparadas"])
    else:
        for col, n in verificacion["diferencias"].items():
            if n:
                log.error("   %s: %s diferencias", col, n)
        raise RuntimeError("El saneamiento del endpoint no reproduce la Fase 1.")

    if verificacion["idempotente"]:
        log.info("   idempotente: sanear dos veces da lo mismo que sanear una")
    else:
        for col, n in verificacion["diferencias_segunda_pasada"].items():
            if n:
                log.error("   segunda pasada, %s: %s diferencias", col, n)
        raise RuntimeError("El saneamiento del endpoint no es idempotente.")

    # --- Artefacto -----------------------------------------------------------
    log.info("-" * 72)
    log.info("PASO 2 | Artefacto de despliegue")
    artefacto = construir_artefacto()
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    with open(RUTA_ARTEFACTO, "w", encoding="utf-8") as f:
        json.dump(artefacto, f, indent=2, ensure_ascii=False)
    log.info("   modelo %s | umbral %.4f | %s caracteristicas",
             artefacto["modelo"]["version"], artefacto["umbral"]["valor"],
             artefacto["modelo"]["n_caracteristicas"])
    for banda in artefacto["bandas"]:
        log.info("   banda %s: %s de train, mora %.2f%%", banda["banda"],
                 banda["n_train"], banda["tasa_mora_train"] * 100)
    log.info("Guardado: %s", RUTA_ARTEFACTO.relative_to(RUTA_RAIZ))
    limpiar_cache()  # el artefacto acaba de cambiar en disco

    # --- Scorecard como artefacto de despliegue ------------------------------
    log.info("-" * 72)
    log.info("PASO 3 | Scorecard publicado")
    scorecard = cargar_scorecard()
    pd.DataFrame(scorecard["tabla"]).to_csv(
        RUTA_SCORECARD, sep=fe.SEPARADOR, encoding=fe.ENCODING, index=False)
    log.info("   %s tramos, %s ajustes continuos, puntos base %s",
             len(scorecard["tabla"]), len(scorecard["ajustes_continuos"]),
             scorecard["puntos_base"])
    log.info("Guardado: %s", RUTA_SCORECARD.relative_to(RUTA_RAIZ))

    # --- Equivalencia con la ruta evaluada -----------------------------------
    #
    # La prueba definitiva contra el train-serving skew: partir del archivo
    # CRUDO y atravesar la cadena del endpoint debe dar exactamente la misma
    # probabilidad que la ruta de evaluacion, que parte del dataset ya limpio.
    log.info("-" * 72)
    log.info("PASO 4 | Equivalencia entre la cadena del endpoint y la evaluada")
    crudo = pd.read_csv(RUTA_RAIZ / fe.CONFIG["data"]["raw_path"],
                        sep=fe.SEPARADOR, encoding=fe.ENCODING)
    limpio = fe.cargar_datos()

    cadena = cadena_inferencia()
    p_endpoint = cadena.predict_proba(sanear(crudo))[:, 1]
    p_evaluacion = cadena.predict_proba(limpio.drop(columns=[TARGET]))[:, 1]
    error = float(np.abs(p_endpoint - p_evaluacion).max())
    log.info("   %s registros | error maximo en probabilidad: %.2e",
             len(crudo), error)
    if error > 1e-12:
        raise RuntimeError("La cadena del endpoint no reproduce la evaluada.")

    # --- Prueba de humo ------------------------------------------------------
    log.info("-" * 72)
    log.info("PASO 5 | Prueba de humo sobre un lote crudo")
    RUTA_MONITOREO.mkdir(parents=True, exist_ok=True)
    lote = crudo[COLUMNAS_REQUERIDAS].sample(50, random_state=fe.SEMILLA)
    lote.to_csv(RUTA_LOTE_EJEMPLO, sep=fe.SEPARADOR, encoding=fe.ENCODING,
                index=False)
    log.info("Guardado: %s", RUTA_LOTE_EJEMPLO.relative_to(RUTA_RAIZ))

    calidad = calidad_del_lote(lote)
    log.info("   calidad de la fuente: %s",
             f"{len(calidad)} avisos" if calidad else "sin avisos")

    resultado = predecir_lote(lote, origen="prueba_de_humo")
    reparto = resultado["decision"].value_counts()
    for decision, n in reparto.items():
        log.info("   %-16s %2s solicitudes (%.0f%%)", decision, n,
                 n / len(resultado) * 100)
    log.info("   puntaje: min %.0f | medio %.0f | max %.0f",
             resultado["puntaje"].min(), resultado["puntaje"].mean(),
             resultado["puntaje"].max())
    log.info("   ejemplo de motivo: %s", resultado["motivo"].iloc[0])
    log.info("Guardado: %s (%s filas)", RUTA_REGISTRO.relative_to(RUTA_RAIZ),
             len(pd.read_csv(RUTA_REGISTRO, sep=fe.SEPARADOR,
                             encoding=fe.ENCODING)))

    log.info("=" * 72)
    log.info("Levantar la app:  uvicorn app:app --app-dir mlops_pipeline --port 8000")
    log.info("=" * 72)

    return {"verificacion": verificacion, "artefacto": artefacto,
            "error_equivalencia": error,
            "prueba_de_humo": reparto.to_dict()}


if __name__ == "__main__":
    main()
