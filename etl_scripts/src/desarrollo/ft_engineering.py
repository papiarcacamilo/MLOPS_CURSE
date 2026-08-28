"""
Fase 2 - Feature Engineering
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : Base_de_datos_limpia.csv 


    Paso 1  Carga y configuracion
    Paso 2  Exclusion de variables con fuga de informacion confirmada
    Paso 3  Particion de datos (estratificada y temporal)

Justificacion del orden (particion antes que transformacion)
--------------------------------------------------------------------------------
Si los cortes de binning, los valores WoE o los parametros de escalado se
calculan sobre el dataset completo, el conjunto de prueba influye en la
transformacion y deja de ser una medida independiente del desempenio. Es una
forma silenciosa de fuga: las metricas salen mejores de lo que seran en
produccion. Por eso la particion es el primer paso ejecutable de la Fase 2.


================================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# CONFIGURACION


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
    en etl_scripts/src/desarrollo/ y los datos en la raiz del repositorio.
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
# `saldo_mora` y `saldo_mora_codeudor` se registran
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

# Columnas auxiliares: se conservan en la particion pero NO son predictoras.

COLUMNAS_AUXILIARES = [
    COLUMNA_FECHA,          # necesaria para el split temporal y la Fase 5
    "anio_mes_prestamo",    # cohorte de desembolso
    "vencimiento_teorico",  # base del calculo de madurez
    "madurez_incompleta",   # decision pendiente para la Fase 3
]

PROPORCION_TEST = 0.20



# CARGA


def cargar_datos() -> pd.DataFrame:
    """Carga el dataset limpio producido por la Fase 1 """
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



# EXCLUSION DE VARIABLES CON FUGA


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
#  PARTICION DE DATOS


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



# VALIDACION


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



# PERSISTENCIA


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



#  ATRIBUTOS DERIVADOS

MIN_POBLACION_BIN = 0.05   # 5% de la poblacion de train
MIN_EVENTOS_BIN = 20       # minimo de casos de mora para estimar un WoE estable
SUAVIZADO = 0.5            # correccion de Laplace en el calculo del WoE


def construir_derivadas(df: pd.DataFrame, fecha_corte_obs: pd.Timestamp) -> pd.DataFrame:
    """Construye los atributos derivados validados en la Fase 1.

    `fecha_corte_obs` es la ultima fecha observada en el dataset COMPLETO y se
    pasa como parametro en lugar de recalcularse: si se derivara de cada
    particion, train y test usarian referencias distintas y la antiguedad no
    seria comparable entre ambos.
    """
    df = df.copy()

    # IV = 0.1637 en la Fase 1, el derivado mas predictivo y por encima de la
    # variable original de la que procede (huella_consulta, IV = 0.1455).
    # El +1 evita la division por cero en clientes sin creditos vigentes.
    df["consultas_por_credito"] = df["huella_consulta"] / (df["cant_creditosvigentes"] + 1)

    # IV = 0.0930. Brecha entre el ingreso declarado y el reportado por el buro.
    # Queda nula donde no hay reporte del buro; el binning la trata como SIN_DATO.
    df["discrepancia_ingresos"] = (
        df["salario_cliente"] - df["promedio_ingresos_datacredito"]
    ).abs()

    # Componentes temporales: la Fase 1 mostro que la tasa de mora varia entre
    # 1.72% y 9.09% segun el mes de desembolso.
    df["antiguedad_dias"] = (fecha_corte_obs - df[COLUMNA_FECHA]).dt.days
    df["mes_prestamo"] = df[COLUMNA_FECHA].dt.month
    df["trimestre_prestamo"] = df[COLUMNA_FECHA].dt.quarter

    # Los tipos 7 y 68 tienen 2 y 1 registro en el dataset completo: como
    # categorias propias no son estimables. El 6 se conserva separado por su
    # tasa del 42.86% (OR ~15, p < 0.0001).
    df["tipo_credito_grp"] = df["tipo_credito"].astype(str).where(
        df["tipo_credito"].isin([4, 6, 9, 10]), "Otros"
    )
    return df



# BINNING Y WEIGHT APLICADO 

# se usa WoE en lugar de One-Hot porque el dominio es
# riesgo crediticio, donde la interpretabilidad es un requisito regulatorio (hay
# que justificar la negacion de un credito ante el cliente y ante el supervisor).
# Ademas resuelve los nulos sin imputar: la ausencia de dato se trata como una
# categoria mas, con su propio WoE estimado a partir de su tasa observada.


# Cortes iniciales derivados de los hallazgos del EDA. La fusion automatica
# posterior los ajusta si algun tramo resulta demasiado pequenio.
CORTES_INICIALES = {
    "puntaje_datacredito": [280, 600, 700, 750, 800, 850, 950],
    "plazo_meses": [0, 6, 12, 18, 24, 36, 90],
}

# Variables numericas que se discretizan por cuantiles (sin cortes de negocio).
VARS_CUANTILES = [
    "huella_consulta",
    "consultas_por_credito",
    "promedio_ingresos_datacredito",
    "discrepancia_ingresos",
    "edad_cliente",
    "cant_creditosvigentes",
]

# Categoricas: reciben WoE directamente sobre sus niveles.
VARS_CATEGORICAS_WOE = ["tipo_laboral", "tendencia_ingresos", "tipo_credito_grp"]

# Monetarias muy asimetricas: log1p + escalado robusto (paso 6).
VARS_MONETARIAS = [
    "capital_prestado",
    "cuota_pactada",
    "salario_cliente",
    "total_otros_prestamos",
]

# Binarias que ya estan en 0/1 y pasan sin transformar.
VARS_BINARIAS = [
    "promedio_ingresos_datacredito_era_nulo",
    "sin_historial_crediticio",
    "edad_cliente_corregida",
    "salario_cliente_corregido",
    "tendencia_ingresos_reconstruida",
    "total_otros_prestamos_sospechoso",
]

ETIQUETA_NULO = "SIN_DATO"


def _discretizar(serie: pd.Series, cortes: list | None,
                 n_cuantiles: int = 5) -> tuple[pd.Series, list]:
    """Convierte una serie numerica en etiquetas de tramo, con SIN_DATO aparte."""
    if cortes is None:
        _, cortes = pd.qcut(serie, n_cuantiles, duplicates="drop", retbins=True)
        cortes = [float(c) for c in cortes]
        cortes[0], cortes[-1] = -np.inf, np.inf
    tramos = pd.cut(serie, cortes).astype(str)
    return tramos.where(serie.notna(), ETIQUETA_NULO), cortes


def _fusionar_bins_pequenos(tramos: pd.Series, objetivo: pd.Series,
                            n_total: int) -> dict:
    """Agrupa los tramos que no alcanzan el tamanio minimo con su vecino.

    Un tramo con muy pocos registros produce un WoE inestable: el modelo aprende
    ruido que no se reproduce en produccion. Caso concreto de este proyecto: la
    banda de score por debajo de 600 puntos tiene 25 registros en el dataset
    completo y una tasa de mora del 64%, lo que generaria un WoE de +3.57
    apoyado en 16 eventos. Se fusiona con la banda contigua.

    La categoria SIN_DATO nunca se fusiona: representa ausencia de informacion,
    no un valor bajo o alto, y mezclarla con un tramo numerico destruiria su
    interpretacion.
    """
    resumen = pd.DataFrame({"tramo": tramos, "mora": objetivo}).groupby(
        "tramo", observed=True
    ).agg(n=("mora", "size"), eventos=("mora", "sum"))

    # Orden natural de los intervalos (SIN_DATO al final, fuera de la fusion)
    numericos = [t for t in resumen.index if t != ETIQUETA_NULO]
    numericos.sort(key=lambda s: float(s.split(",")[0].strip("([ ")))

    min_n = max(int(n_total * MIN_POBLACION_BIN), 1)
    grupos, actual = [], []
    for tramo in numericos:
        actual.append(tramo)
        n = resumen.loc[actual, "n"].sum()
        ev = resumen.loc[actual, "eventos"].sum()
        if n >= min_n or ev >= MIN_EVENTOS_BIN:
            grupos.append(list(actual))
            actual = []
    if actual:                       # el resto se une al ultimo grupo cerrado
        if grupos:
            grupos[-1].extend(actual)
        else:
            grupos.append(list(actual))

    mapa = {}
    for grupo in grupos:
        etiqueta = grupo[0] if len(grupo) == 1 else (
            grupo[0].split(",")[0] + ", " + grupo[-1].split(",")[1]
        )
        for tramo in grupo:
            mapa[tramo] = etiqueta
    if ETIQUETA_NULO in resumen.index:
        mapa[ETIQUETA_NULO] = ETIQUETA_NULO
    return mapa


def ajustar_woe(tramos: pd.Series, objetivo: pd.Series) -> tuple[dict, float]:
    """Calcula el WoE de cada tramo y el Information Value de la variable.

    WoE = ln( %eventos_del_tramo / %no_eventos_del_tramo ). Un WoE positivo
    indica mas riesgo que la media de la cartera. Se aplica suavizado de Laplace
    para evitar divisiones por cero en tramos sin eventos.
    """
    tabla = pd.crosstab(tramos, objetivo)
    for clase in (0, 1):
        if clase not in tabla.columns:
            tabla[clase] = 0
    eventos = (tabla[1] + SUAVIZADO) / (tabla[1].sum() + SUAVIZADO * len(tabla))
    no_eventos = (tabla[0] + SUAVIZADO) / (tabla[0].sum() + SUAVIZADO * len(tabla))
    woe = np.log(eventos / no_eventos)
    iv = float(((eventos - no_eventos) * woe).sum())
    return woe.round(6).to_dict(), round(iv, 4)


def ajustar_binning(train: pd.DataFrame, objetivo: pd.Series) -> dict:
    """Aprende cortes, fusiones y valores WoE. SOLO se ejecuta sobre train."""
    receta = {}
    n = len(train)

    for col in list(CORTES_INICIALES) + VARS_CUANTILES:
        cortes = CORTES_INICIALES.get(col)
        tramos, cortes = _discretizar(train[col], cortes)
        mapa = _fusionar_bins_pequenos(tramos, objetivo, n)
        tramos = tramos.map(mapa)
        woe, iv = ajustar_woe(tramos, objetivo)
        receta[col] = {"tipo": "numerica", "cortes": cortes,
                       "fusion": mapa, "woe": woe, "iv": iv}

    for col in VARS_CATEGORICAS_WOE:
        tramos = train[col].astype(str)
        woe, iv = ajustar_woe(tramos, objetivo)
        receta[col] = {"tipo": "categorica", "woe": woe, "iv": iv}

    return receta


def aplicar_binning(df: pd.DataFrame, receta: dict) -> pd.DataFrame:
    """Aplica la receta aprendida en train. No recalcula ningun estadistico.

    Los niveles no vistos en train reciben WoE = 0 (riesgo igual al promedio),
    que es la decision conservadora: no inventa una senial que no se aprendio.
    """
    salida = pd.DataFrame(index=df.index)
    for col, spec in receta.items():
        if spec["tipo"] == "numerica":
            tramos, _ = _discretizar(df[col], spec["cortes"])
            tramos = tramos.map(lambda t: spec["fusion"].get(t, t))
        else:
            tramos = df[col].astype(str)
        salida[f"woe_{col}"] = tramos.map(spec["woe"]).fillna(0.0).astype(float)
    return salida



#  CODIFICACION Y ESCALADO


def ajustar_escalado(train: pd.DataFrame) -> dict:
    """Aprende mediana e IQR de las variables monetarias. Solo sobre train.

    Se usa escalado robusto (mediana e IQR) en lugar de estandarizacion porque
    la Fase 1 midio asimetrias de 2.2 a 38.5 en estas variables: la media y la
    desviacion estandar quedan dominadas por la cola derecha. Se aplica log1p
    antes para comprimir esa cola.
    """
    parametros = {}
    for col in VARS_MONETARIAS:
        serie = np.log1p(train[col])
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        parametros[col] = {"mediana": float(serie.median()),
                           "iqr": float(iqr if iqr > 0 else 1.0)}
    return parametros


def aplicar_escalado(df: pd.DataFrame, parametros: dict) -> pd.DataFrame:
    """Aplica log1p y el escalado robusto aprendido en train."""
    salida = pd.DataFrame(index=df.index)
    for col, p in parametros.items():
        salida[f"esc_{col}"] = (np.log1p(df[col]) - p["mediana"]) / p["iqr"]
    return salida


def ensamblar_matriz(df: pd.DataFrame, receta: dict, parametros: dict) -> pd.DataFrame:
    """Construye la matriz final de caracteristicas lista para modelar."""
    partes = [
        aplicar_binning(df, receta),
        aplicar_escalado(df, parametros),
        df[[c for c in VARS_BINARIAS if c in df.columns]].astype(int),
        df[["mes_prestamo", "trimestre_prestamo", "antiguedad_dias"]],
        # `madurez_incompleta` entra como caracteristica: la Fase 1 la valido
        # como significativa (6.40% vs 4.34%, p = 0.0001).
        df[["madurez_incompleta"]].astype(int),
    ]
    matriz = pd.concat(partes, axis=1)
    matriz[TARGET] = df[TARGET].values
    return matriz


#
# VALIDACION DEL APORTE DE LAS CARACTERISTICAS


def _iv_simple(serie: pd.Series, objetivo: pd.Series, bins: int = 10) -> float:
    """IV de una variable discretizada en `bins` tramos, sin fusion ni suavizado fuerte.

    Se usa para dos comparaciones distintas:
      - con bins=10: IV "bruto", util como referencia de cuanta senial hay en la
        variable si se le permite sobreajustar con muchos tramos.
      - con bins = n_tramos de la version final: comparacion JUSTA, porque el IV
        crece mecanicamente con el numero de tramos y comparar 10 contra 4
        penalizaria a la version regularizada sin motivo real.
    """
    if serie.dtype.kind in "ifc" and serie.nunique() > bins:
        try:
            tramos = pd.qcut(serie, bins, duplicates="drop").astype(str)
        except ValueError:
            return float("nan")
    else:
        tramos = serie.astype(str)
    tramos = tramos.where(serie.notna(), ETIQUETA_NULO)
    _, iv = ajustar_woe(tramos, objetivo)
    return iv


def validar_features(train: pd.DataFrame, receta: dict, objetivo: pd.Series) -> dict:
    """Compara IV antes/despues, revisa monotonia y busca indicios de fuga."""
    filas, alertas = [], []
    for col, spec in receta.items():
        n_tramos = len(spec["woe"])
        iv_bruto = _iv_simple(train[col], objetivo, bins=10)
        iv_comparable = _iv_simple(train[col], objetivo, bins=n_tramos)
        iv_despues = spec["iv"]

        monotono = None
        if spec["tipo"] == "numerica":
            orden = [k for k in spec["woe"] if k != ETIQUETA_NULO]
            orden.sort(key=lambda s: float(s.split(",")[0].strip("([ ")))
            valores = [spec["woe"][k] for k in orden]
            difs = np.diff(valores)
            monotono = bool(np.all(difs >= 0) or np.all(difs <= 0))

        if iv_despues > 0.5:
            alertas.append(f"{col}: IV={iv_despues} > 0.5, revisar posible fuga")

        if iv_despues < 0.02:
            alertas.append(f"{col}: IV={iv_despues} < 0.02, sin poder predictivo")

        filas.append({
            "variable": col,
            "iv_bruto_10tramos": iv_bruto,
            "iv_comparable": iv_comparable,      # mismo n de tramos que la final
            "iv_transformada": iv_despues,
            "vs_comparable": round(iv_despues - (iv_comparable or 0), 4),
            "retencion_vs_bruto": round(iv_despues / iv_bruto, 3) if iv_bruto else None,
            "n_tramos": n_tramos,
            "woe_monotono": monotono,
        })

    ranking = pd.DataFrame(filas).sort_values("iv_transformada", ascending=False)
    return {"ranking": ranking, "alertas": alertas}


def baseline_comparativo(matriz_train: pd.DataFrame,
                         crudo_train: pd.DataFrame | None = None) -> dict:
    """Compara features transformadas vs originales con regresion logistica.

    Se usa AUC-PR (average precision) sobre la clase minoritaria en validacion
    cruzada estratificada: con 4.75% de eventos, el AUC-ROC resulta
    excesivamente optimista y la exactitud es inservible.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    def evaluar(X, y):
        # El escalado va DENTRO del pipeline para que se ajuste en cada fold de
        # entrenamiento y no en el fold de validacion: hacerlo fuera seria fuga.
        modelo = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEMILLA),
        )
        s = cross_val_score(modelo, X, y, cv=cv, scoring="average_precision")
        return round(float(s.mean()), 4), round(float(s.std()), 4), \
            [round(float(v), 4) for v in s]

    y = 1 - matriz_train[TARGET]
    media, desv, folds = evaluar(matriz_train.drop(columns=[TARGET]), y)
    tasa_base = round(float(y.mean()), 4)

    resultado = {
        "transformadas": {"auc_pr_media": media, "auc_pr_desv": desv, "folds": folds},
        "tasa_base": tasa_base,
        "lift_vs_azar": round(media / tasa_base, 2),
    }

    # Comparacion contra las variables originales sin transformar: es la prueba
    # de que el trabajo de la Fase 2 aporta y no solo reordena informacion.
    if crudo_train is not None:
        X0 = crudo_train.select_dtypes(include=[np.number]).drop(
            columns=[TARGET], errors="ignore")
        m0, d0, f0 = evaluar(X0, y)
        resultado["originales"] = {"auc_pr_media": m0, "auc_pr_desv": d0,
                                   "folds": f0, "n_features": X0.shape[1]}
        resultado["ganancia_abs"] = round(media - m0, 4)
        resultado["ganancia_rel"] = round((media - m0) / m0, 3) if m0 else None

    return resultado

def guardar_recetas(reportes: dict) -> None:
    """Persiste la receta de transformacion. Es el artefacto reutilizable.

    En la Fase 4 el modelo en produccion debe aplicar EXACTAMENTE los mismos
    cortes y valores WoE aprendidos aqui. Guardarlos en disco evita que se
    recalculen sobre datos nuevos, que seria una fuente de deriva silenciosa.
    """
    for nombre, rep in reportes.items():
        destino = RUTA_SALIDA / f"receta_{nombre}.json"
        with open(destino, "w", encoding="utf-8") as f:
            json.dump({"binning_woe": rep["receta"], "escalado": rep["escalado"],
                       "n_features": rep["n_features"]}, f, indent=2,
                      ensure_ascii=False)
        log.info("Guardado: data/processed/receta_%s.json", nombre)

    resumen = {n: {"ranking_iv": r["ranking_iv"], "alertas": r["alertas"],
                   "baseline": r["baseline"], "n_features": r["n_features"]}
               for n, r in reportes.items()}
    with open(RUTA_SALIDA / "reporte_features.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    log.info("Guardado: data/processed/reporte_features.json")



# ORQUESTACION DE LA ETAPA


def main() -> dict:
    log.info("=" * 72)
    log.info("FASE 2 - FEATURE ENGINEERING | Pasos 1-7")
    log.info("=" * 72)

    
    df = cargar_datos()
    fecha_corte_obs = df[COLUMNA_FECHA].max()
    df = excluir_variables_con_fuga(df)

  
    log.info("-" * 72)
    particiones, metricas = {}, {}
    tr_e, te_e = split_estratificado(df)
    metricas["estratificada"] = validar_particion("estratificado", tr_e, te_e, len(df))
    tr_t, te_t, corte = split_temporal(df)
    metricas["temporal"] = validar_particion("temporal", tr_t, te_t, len(df))
    log.info("[temporal] fecha de corte: %s", corte.date())
    log.warning("[temporal] madurez incompleta: train %.1f%% vs test %.1f%% "
                "-> la tasa de mora del test subestima la real",
                metricas["temporal"]["madurez_incompleta_train_pct"],
                metricas["temporal"]["madurez_incompleta_test_pct"])

    conjuntos = {"estratificado": (tr_e, te_e), "temporal": (tr_t, te_t)}

    
    log.info("-" * 72)
    log.info("PASO 4 | Atributos derivados (operaciones fila a fila, sin fuga)")
    for nombre, (tr, te) in conjuntos.items():
        conjuntos[nombre] = (construir_derivadas(tr, fecha_corte_obs),
                             construir_derivadas(te, fecha_corte_obs))
    nuevas = ["consultas_por_credito", "discrepancia_ingresos", "antiguedad_dias",
              "mes_prestamo", "trimestre_prestamo", "tipo_credito_grp"]
    log.info("Construidos: %s", nuevas)

  
    reportes = {}
    for nombre, (tr, te) in conjuntos.items():
        log.info("-" * 72)
        log.info("PASOS 5-6 | Particion '%s': ajuste sobre TRAIN, aplicacion a TEST",
                 nombre)
        y_train = 1 - tr[TARGET]

        receta = ajustar_binning(tr, y_train)
        parametros = ajustar_escalado(tr)

        m_tr = ensamblar_matriz(tr, receta, parametros)
        m_te = ensamblar_matriz(te, receta, parametros)
        log.info("Matriz de caracteristicas: train %s x %s | test %s x %s",
                 *m_tr.shape, *m_te.shape)

        log.info("PASO 7 | Validacion del aporte de las caracteristicas")
        val = validar_features(tr, receta, y_train)
        for a in val["alertas"]:
            log.warning("ALERTA %s", a)
        rk = val["ranking"]
        log.info("IV: %s de %s variables igualan o superan su version con el mismo "
                 "numero de tramos | retencion media vs 10 tramos: %.0f%%",
                 int((rk["vs_comparable"] >= 0).sum()), len(rk),
                 rk["retencion_vs_bruto"].dropna().mean() * 100)
        no_mono = rk[rk["woe_monotono"] == False]["variable"].tolist()
        if no_mono:
            log.info("WoE no monotono (esperado en %s): %s",
                     "plazo_meses por el hallazgo del EDA", no_mono)

        base = baseline_comparativo(m_tr, crudo_train=tr)
        log.info("Baseline 5-fold | transformadas AUC-PR = %.4f +/- %.4f "
                 "| originales = %.4f | tasa base = %.4f",
                 base["transformadas"]["auc_pr_media"],
                 base["transformadas"]["auc_pr_desv"],
                 base.get("originales", {}).get("auc_pr_media", float("nan")),
                 base["tasa_base"])
        log.info("Lift sobre el azar: %.2fx | ganancia vs originales: %+.4f (%+.1f%%)",
                 base["lift_vs_azar"], base.get("ganancia_abs", 0),
                 (base.get("ganancia_rel") or 0) * 100)
        if base["transformadas"]["auc_pr_media"] <= base["tasa_base"]:
            log.error("El baseline no supera la tasa base: las features no aportan")
        if base.get("ganancia_abs", 0) < 0:
            log.warning("Las features transformadas NO superan a las originales")

        particiones[f"{nombre}_train_features"] = m_tr
        particiones[f"{nombre}_test_features"] = m_te
        reportes[nombre] = {
            "receta": receta,
            "escalado": parametros,
            "ranking_iv": val["ranking"].to_dict(orient="records"),
            "alertas": val["alertas"],
            "baseline": base,
            "n_features": m_tr.shape[1] - 1,
        }

    # ------------------- Persistencia -------------
    log.info("-" * 72)
    for nombre, (tr, te) in conjuntos.items():
        particiones[f"{nombre}_train"] = tr
        particiones[f"{nombre}_test"] = te

    metadatos = {
        "generado_por": "ft_engineering.py (Fase 2, pasos 1-7)",
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
        "atributos_derivados": nuevas,
        "criterio_binning": {
            "min_poblacion_bin": MIN_POBLACION_BIN,
            "min_eventos_bin": MIN_EVENTOS_BIN,
            "suavizado_laplace": SUAVIZADO,
            "nota": ("Los tramos que no alcanzan el minimo se fusionan con el "
                     "contiguo. La categoria SIN_DATO nunca se fusiona: la "
                     "ausencia de dato no es un valor alto ni bajo."),
        },
        "columnas_auxiliares_no_predictoras": COLUMNAS_AUXILIARES,
        "particion_estratificada": metricas["estratificada"],
        "particion_temporal": {**metricas["temporal"], "fecha_corte": str(corte.date())},
        "advertencia_madurez": (
            "En la particion temporal, el conjunto de prueba concentra creditos "
            "recientes cuyo plazo aun no ha vencido. Su tasa de mora observada "
            "subestima la real. Evaluar en la Fase 3 con y sin esos registros."
        ),
    }

    guardar_particiones(particiones, metadatos)
    guardar_recetas(reportes)

    log.info("=" * 72)
    log.info("FASE 2 COMPLETADA. Salida en data/processed/")
    log.info("=" * 72)
    return {"metadatos": metadatos, "reportes": reportes}


if __name__ == "__main__":
    main()
