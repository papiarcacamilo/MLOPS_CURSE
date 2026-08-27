"""
Fase 2 - Feature Engineering
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : Base_de_datos_limpia.csv (salida verificada de la Fase 1)
Alcance  : ESTE MODULO CUBRE UNICAMENTE LOS PASOS 1 A 3 DEL PLAN DE LA FASE 2.

    Paso 1  Carga y configuracion
    Paso 2  Exclusion de variables con fuga de informacion confirmada
    Paso 3  Particion de datos (estratificada y temporal)

    NO se implementa aqui: atributos derivados, binning, WoE, codificacion ni
    escalado. Esas etapas son posteriores y dependen de que la particion ya
    exista, porque todo estadistico de transformacion debe ajustarse UNICAMENTE
    sobre el conjunto de entrenamiento.

Justificacion del orden (particion antes que transformacion)
--------------------------------------------------------------------------------
Si los cortes de binning, los valores WoE o los parametros de escalado se
calculan sobre el dataset completo, el conjunto de prueba influye en la
transformacion y deja de ser una medida independiente del desempenio. Es una
forma silenciosa de fuga: las metricas salen mejores de lo que seran en
produccion. Por eso la particion es el primer paso ejecutable de la Fase 2.

Autor    : Proyecto academico - Ciencia de Datos en Produccion
================================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURACION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ft_engineering")


def encontrar_raiz(marcador: str = "Base_de_datos.csv", max_niveles: int = 6) -> Path:
    """Sube por el arbol de directorios hasta encontrar el archivo marcador.

    Se replica la estrategia usada en `transformacion_eda.ipynb`: el script vive
    en etl_scripts/src/desarrollo/ y los datos en la raiz del repositorio. Fijar
    '../../../' a mano se rompe si el archivo se mueve o si el interprete arranca
    desde otro directorio de trabajo.
    """
    try:
        actual = Path(__file__).resolve().parent
    except NameError:
        actual = Path.cwd().resolve()
    for _ in range(max_niveles + 1):
        if (actual / marcador).exists():
            return actual
        if actual.parent == actual:
            break
        actual = actual.parent
    raise FileNotFoundError(
        f"No se encontro '{marcador}' subiendo desde {actual}. "
        "Verifica que el script este dentro del repositorio del proyecto."
    )


RUTA_RAIZ = encontrar_raiz()
RUTA_CONFIG = RUTA_RAIZ / "etl_scripts" / "src" / "config.json"
RUTA_SALIDA = RUTA_RAIZ / "data" / "processed"


def cargar_config(ruta: Path = RUTA_CONFIG) -> dict:
    """Lee config.json. Centraliza semilla, separador y nombre del target."""
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    log.info("Configuracion cargada desde %s", ruta.relative_to(RUTA_RAIZ))
    return cfg


CONFIG = cargar_config()
TARGET = CONFIG["target_variable"]
SEMILLA = CONFIG["random_state"]
SEPARADOR = CONFIG["data"]["separator"]
ENCODING = CONFIG["data"]["encoding"]
COLUMNA_FECHA = "fecha_prestamo"

# ------------------------------------------------------------------------------
# Variables excluidas por fuga de informacion
# ------------------------------------------------------------------------------
# El negocio confirmo que `saldo_mora` y `saldo_mora_codeudor` se registran
# DESPUES del desembolso. Son consecuencia y no causa del impago, y ademas el
# dato no existe en el momento de originar el credito.
FUGA_CONFIRMADA = [
    "saldo_mora",
    "saldo_mora_codeudor",
    "saldo_mora_era_nulo",
    "saldo_mora_codeudor_era_nulo",
]

# Provienen del mismo corte de informacion del buro que las anteriores, por lo
# que se excluyen por precaucion. Su Information Value medido en la Fase 1 esta
# entre 0.018 y 0.021 (sin poder predictivo), asi que la exclusion no tiene coste.
FUGA_PRECAUCION = [
    "saldo_total",
    "saldo_principal",
    "saldo_total_era_nulo",
    "saldo_principal_era_nulo",
]

VARIABLES_EXCLUIDAS = FUGA_CONFIRMADA + FUGA_PRECAUCION

# `tiene_mora_previa` (saldo_mora > 0) fue el atributo derivado con mayor
# correlacion en la Fase 1. NO se construye: deriva de una variable con fuga.

# ------------------------------------------------------------------------------
# Columnas auxiliares: se conservan en la particion pero NO son predictoras.
# Sirven para trazabilidad y para decisiones de la Fase 3.
# ------------------------------------------------------------------------------
COLUMNAS_AUXILIARES = [
    COLUMNA_FECHA,          # necesaria para el split temporal y la Fase 5
    "anio_mes_prestamo",    # cohorte de desembolso
    "vencimiento_teorico",  # base del calculo de madurez
    "madurez_incompleta",   # decision pendiente para la Fase 3
]

PROPORCION_TEST = 0.20


# ==============================================================================
# PASO 1 - CARGA
# ==============================================================================

def cargar_datos() -> pd.DataFrame:
    """Carga el dataset limpio producido por la Fase 1 y verifica su integridad."""
    ruta = RUTA_RAIZ / CONFIG["data"]["clean_path"]
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero el notebook de la Fase 1."
        )

    df = pd.read_csv(ruta, sep=SEPARADOR, encoding=ENCODING)
    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA])
    df["madurez_incompleta"] = df["madurez_incompleta"].astype(str).str.lower().isin(
        ["true", "1"]
    )

    log.info("Dataset cargado: %s registros x %s columnas", *df.shape)

    # Contrato de entrada: si la Fase 1 cambia, este script debe fallar de forma
    # explicita en lugar de producir una particion silenciosamente incorrecta.
    if TARGET not in df.columns:
        raise ValueError(f"Falta la variable objetivo '{TARGET}'.")
    if df[TARGET].isna().any():
        raise ValueError("La variable objetivo contiene nulos.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"'{TARGET}' debe ser binaria (0/1).")
    if df.duplicated().any():
        raise ValueError("El dataset contiene filas duplicadas.")

    tasa = (1 - df[TARGET].mean()) * 100
    log.info("Tasa de mora global: %.2f%% (%s casos)", tasa, int((1 - df[TARGET]).sum()))
    return df


# ==============================================================================
# PASO 2 - EXCLUSION DE VARIABLES CON FUGA
# ==============================================================================

def excluir_variables_con_fuga(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las variables contaminadas por informacion posterior al desembolso."""
    presentes = [c for c in VARIABLES_EXCLUIDAS if c in df.columns]
    ausentes = [c for c in VARIABLES_EXCLUIDAS if c not in df.columns]

    if ausentes:
        log.warning("Variables a excluir no encontradas en el dataset: %s", ausentes)

    df = df.drop(columns=presentes)

    log.info("Excluidas por fuga confirmada : %s", FUGA_CONFIRMADA)
    log.info("Excluidas por precaucion      : %s", FUGA_PRECAUCION)
    log.info("Dataset tras exclusion: %s registros x %s columnas", *df.shape)
    return df


# ==============================================================================
# PASO 3 - PARTICION DE DATOS
# ==============================================================================

def split_estratificado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particion aleatoria conservando la proporcion de clases.

    Con una tasa de mora del 4.75% (ratio ~20:1), una particion aleatoria simple
    puede dejar el conjunto de prueba con una proporcion de positivos muy distinta
    a la real solo por azar. La estratificacion fija esa proporcion en ambos lados.
    """
    train, test = train_test_split(
        df,
        test_size=PROPORCION_TEST,
        stratify=df[TARGET],
        random_state=SEMILLA,
        shuffle=True,
    )
    return train.copy(), test.copy()


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Particion por fecha de desembolso: entrena con el pasado, valida con el futuro.

    La Fase 1 documento que la tasa de mora no es estable entre cohortes mensuales
    (de 1.72% a 9.09%). Un split aleatorio asume implicitamente que todos los
    periodos son intercambiables, supuesto que estos datos no respaldan. Esta
    particion reproduce la situacion real de despliegue: el modelo se entrena con
    creditos historicos y debe predecir sobre creditos nuevos.
    """
    df = df.sort_values(COLUMNA_FECHA).reset_index(drop=True)

    # `fecha_prestamo` incluye hora, por lo que el cuantil cae a mitad de una
    # jornada y parte ese dia entre ambos conjuntos. Se normaliza al inicio del
    # dia para que ninguna fecha calendario quede en los dos lados: un mismo dia
    # de originacion puede compartir lote, analista o campania comercial, y
    # repartirlo entre train y test filtra informacion del periodo de prueba.
    corte = df[COLUMNA_FECHA].quantile(1 - PROPORCION_TEST).normalize()

    train = df[df[COLUMNA_FECHA] < corte].copy()
    test = df[df[COLUMNA_FECHA] >= corte].copy()
    return train, test, corte


# ==============================================================================
# VALIDACION
# ==============================================================================

def validar_particion(nombre: str, train: pd.DataFrame, test: pd.DataFrame,
                      total: int) -> dict:
    """Comprueba integridad de la particion y devuelve sus metricas descriptivas."""
    errores = []

    # 1. Sin perdida ni duplicacion de registros
    if len(train) + len(test) != total:
        errores.append(f"suma {len(train)}+{len(test)} != {total}")

    # 2. Sin solapamiento de indices entre train y test
    solape = set(train.index) & set(test.index)
    if solape:
        errores.append(f"{len(solape)} indices compartidos entre train y test")

    # 3. Ambas clases representadas en los dos conjuntos
    for etiqueta, parte in [("train", train), ("test", test)]:
        if parte[TARGET].nunique() < 2:
            errores.append(f"{etiqueta} no contiene ambas clases")

    # 4. Minimo de casos positivos en test para que las metricas sean estables
    mora_test = int((1 - test[TARGET]).sum())
    if mora_test < 30:
        errores.append(f"solo {mora_test} casos de mora en test (minimo recomendado 30)")

    # 5. En la particion temporal, ningun dia calendario puede estar en ambos lados
    if nombre == "temporal":
        compartidas = set(train[COLUMNA_FECHA].dt.date) & set(test[COLUMNA_FECHA].dt.date)
        if compartidas:
            errores.append(f"{len(compartidas)} fecha(s) presentes en train y test: "
                           f"{sorted(compartidas)}")
        if train[COLUMNA_FECHA].max() >= test[COLUMNA_FECHA].min():
            errores.append("el maximo de train no es anterior al minimo de test")

    if errores:
        for e in errores:
            log.error("[%s] %s", nombre, e)
        raise ValueError(f"La particion '{nombre}' no supero la validacion.")

    metricas = {
        "n_train": len(train),
        "n_test": len(test),
        "pct_test": round(len(test) / total * 100, 2),
        "mora_train_n": int((1 - train[TARGET]).sum()),
        "mora_test_n": mora_test,
        "mora_train_pct": round((1 - train[TARGET].mean()) * 100, 2),
        "mora_test_pct": round((1 - test[TARGET].mean()) * 100, 2),
        "madurez_incompleta_train_pct": round(train["madurez_incompleta"].mean() * 100, 1),
        "madurez_incompleta_test_pct": round(test["madurez_incompleta"].mean() * 100, 1),
    }
    log.info("[%s] train=%s (mora %.2f%%) | test=%s (mora %.2f%%) -> validacion OK",
             nombre, metricas["n_train"], metricas["mora_train_pct"],
             metricas["n_test"], metricas["mora_test_pct"])
    return metricas


# ==============================================================================
# PERSISTENCIA
# ==============================================================================

def guardar_particiones(particiones: dict, metadatos: dict) -> None:
    """Escribe los conjuntos y el archivo de metadatos de la particion."""
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    for nombre, datos in particiones.items():
        destino = RUTA_SALIDA / f"{nombre}.csv"
        datos.to_csv(destino, sep=SEPARADOR, index=False, encoding=ENCODING)
        log.info("Guardado: data/processed/%s.csv (%s filas)", nombre, len(datos))

    ruta_meta = RUTA_SALIDA / "split_metadata.json"
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False)
    log.info("Guardado: data/processed/split_metadata.json")


# ==============================================================================
# ORQUESTACION
# ==============================================================================

def main() -> dict:
    log.info("=" * 72)
    log.info("FASE 2 - FEATURE ENGINEERING | Pasos 1-3: carga, exclusion y particion")
    log.info("=" * 72)

    # --- Paso 1
    df = cargar_datos()

    # --- Paso 2
    df = excluir_variables_con_fuga(df)

    # --- Paso 3
    log.info("-" * 72)
    tr_e, te_e = split_estratificado(df)
    met_e = validar_particion("estratificado", tr_e, te_e, len(df))

    tr_t, te_t, corte = split_temporal(df)
    met_t = validar_particion("temporal", tr_t, te_t, len(df))
    log.info("[temporal] fecha de corte: %s", corte.date())
    log.warning("[temporal] madurez incompleta: train %.1f%% vs test %.1f%% "
                "-> la tasa de mora del test subestima la real",
                met_t["madurez_incompleta_train_pct"],
                met_t["madurez_incompleta_test_pct"])

    # --- Persistencia
    log.info("-" * 72)
    metadatos = {
        "generado_por": "ft_engineering.py (Fase 2, pasos 1-3)",
        "dataset_origen": CONFIG["data"]["clean_path"],
        "n_registros_origen": len(df),
        "variable_objetivo": TARGET,
        "semilla": SEMILLA,
        "proporcion_test": PROPORCION_TEST,
        "variables_excluidas": {
            "fuga_confirmada": FUGA_CONFIRMADA,
            "fuga_precaucion": FUGA_PRECAUCION,
            "atributo_no_construido": "tiene_mora_previa (deriva de saldo_mora)",
        },
        "columnas_auxiliares_no_predictoras": COLUMNAS_AUXILIARES,
        "particion_estratificada": met_e,
        "particion_temporal": {**met_t, "fecha_corte": str(corte.date())},
        "advertencia_madurez": (
            "En la particion temporal, el conjunto de prueba concentra creditos "
            "recientes cuyo plazo aun no ha vencido. Su tasa de mora observada "
            "subestima la real. Evaluar en la Fase 3 con y sin esos registros."
        ),
    }

    guardar_particiones(
        {
            "estratificado_train": tr_e,
            "estratificado_test": te_e,
            "temporal_train": tr_t,
            "temporal_test": te_t,
        },
        metadatos,
    )

    log.info("=" * 72)
    log.info("Pasos 1-3 completados. Siguiente: atributos derivados (paso 4).")
    log.info("=" * 72)
    return metadatos


if __name__ == "__main__":
    main()
