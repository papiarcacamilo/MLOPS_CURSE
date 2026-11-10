"""
Entregable 5, Fase 2 - Tablas de modelado para Vertex AI
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/processed/estratificado_train.csv y estratificado_test.csv
Salida   : dos tablas de BigQuery, una para AutoML y otra para el pipeline propio

La tabla de AutoML sale de las mismas particiones con las que se entreno y
evaluo el modelo de Python, para que la comparacion sea justa:

  1. Mismo test: los 2.153 creditos del test estratificado, con 102 moras.
  2. Sin fuga: las particiones ya vienen sin las variables excluidas.
  3. Misma clase positiva: `mora` vale 1 cuando el credito no se pago a tiempo.
  4. Mismas entradas: las columnas que recibe el endpoint, sin la fecha.

Por que sin la fecha: el modelo de Python no la usa como variable. Con una
particion aleatoria, AutoML podria aprender en que periodo cayo cada credito y
ganar por una informacion que el modelo comparado no tiene.

Por que hay validacion: AutoML necesita un conjunto para detener el
entrenamiento. Sale del train, estratificado y con la misma semilla. El test no
se toca.

La tabla del pipeline lleva las particiones procesadas completas, porque el
modelo de Python se entrena con las columnas derivadas de ft_engineering.py.
Cumple el mismo contrato: el mismo test y ninguna variable con fuga.

Uso:
    python mlops_pipeline/src/vertex_datos.py                    tabla de AutoML
    python mlops_pipeline/src/vertex_datos.py --tabla pipeline   tabla del pipeline
    Con --subir, ademas reemplaza la tabla en BigQuery.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
RUTA_PROCESADOS = RAIZ / "data" / "processed"
RUTA_ARTEFACTO = RAIZ / "data" / "models" / "artefacto_despliegue.json"

COLUMNA_FECHA = "fecha_prestamo"
OBJETIVO = "mora"
PARTICION = "particion"
ID = "id_fila"

# Contrato de la comparacion: si las particiones cambian, la tabla no se sube.
TEST_ESPERADO = 2153
MORAS_TEST_ESPERADAS = 102
TRAIN_ESPERADO = 8610


def cargar_config(raiz: Path = RAIZ) -> dict:
    with open(raiz / "config.json", encoding="utf-8") as f:
        return json.load(f)


def leer_particion(nombre: str, config: dict, carpeta: Path = RUTA_PROCESADOS) -> pd.DataFrame:
    datos = config["data"]
    return pd.read_csv(carpeta / f"{nombre}.csv", sep=datos["separator"],
                       encoding=datos["encoding"])


def entradas_del_modelo(ruta: Path = RUTA_ARTEFACTO) -> list[str]:
    """Columnas que recibe el endpoint del modelo servido, sin la fecha."""
    with open(ruta, encoding="utf-8") as f:
        artefacto = json.load(f)
    columnas = artefacto["esquema_entrada"]["columnas_requeridas"]
    return [c for c in columnas if c != COLUMNA_FECHA]


def construir_tabla(train: pd.DataFrame, test: pd.DataFrame, entradas: list[str],
                    objetivo_origen: str, semilla: int,
                    proporcion_validacion: float) -> pd.DataFrame:
    """Une train y test con una columna de particion para AutoML."""
    def preparar(df: pd.DataFrame, prefijo: str) -> pd.DataFrame:
        tabla = df[entradas].copy()
        tabla[OBJETIVO] = (1 - df[objetivo_origen]).astype(int)
        tabla.insert(0, ID, [f"{prefijo}{i:05d}" for i in range(len(df))])
        return tabla

    entrenamiento = preparar(train, "tr")
    prueba = preparar(test, "te")

    _, validacion = train_test_split(
        entrenamiento,
        test_size=proporcion_validacion,
        stratify=entrenamiento[OBJETIVO],
        random_state=semilla,
    )
    entrenamiento[PARTICION] = "TRAIN"
    entrenamiento.loc[validacion.index, PARTICION] = "VALIDATE"
    prueba[PARTICION] = "TEST"
    return pd.concat([entrenamiento, prueba], ignore_index=True)


def verificar_tabla(tabla: pd.DataFrame, entradas: list[str]) -> list[str]:
    """Devuelve los problemas encontrados. Una lista vacia aprueba la tabla."""
    problemas = []
    esperadas = [ID, *entradas, OBJETIVO, PARTICION]
    if list(tabla.columns) != esperadas:
        problemas.append(f"columnas inesperadas: {list(tabla.columns)}")
        return problemas

    if not set(tabla[PARTICION]) <= {"TRAIN", "VALIDATE", "TEST"}:
        problemas.append(f"valores de particion no validos: {sorted(set(tabla[PARTICION]))}")
    if not set(tabla[OBJETIVO]) <= {0, 1}:
        problemas.append("el objetivo no es binario")
    if tabla[ID].duplicated().any():
        problemas.append("hay identificadores repetidos")

    test = tabla[tabla[PARTICION] == "TEST"]
    if len(test) != TEST_ESPERADO or int(test[OBJETIVO].sum()) != MORAS_TEST_ESPERADAS:
        problemas.append(f"el test no es el de Python: {len(test)} creditos y "
                         f"{int(test[OBJETIVO].sum())} moras; se esperaban "
                         f"{TEST_ESPERADO} y {MORAS_TEST_ESPERADAS}")
    if (tabla[PARTICION] != "TEST").sum() != TRAIN_ESPERADO:
        problemas.append("train y validacion no suman el train de Python")
    validacion = tabla[tabla[PARTICION] == "VALIDATE"]
    if validacion.empty or validacion[OBJETIVO].sum() == 0:
        problemas.append("la validacion no tiene moras")
    return problemas


def variables_excluidas(carpeta: Path = RUTA_PROCESADOS) -> list[str]:
    """Las variables con fuga que la particion ya excluyo, segun su metadata."""
    with open(carpeta / "split_metadata.json", encoding="utf-8") as f:
        excluidas = json.load(f)["variables_excluidas"]
    return [*excluidas["fuga_confirmada"], *excluidas["fuga_precaucion"]]


def construir_tabla_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Las particiones procesadas tal cual, para reentrenar el modelo de Python.

    El modelo no usa las 12 entradas crudas de AutoML sino las columnas derivadas
    de ft_engineering.py. Aqui no hay validacion: el modelo se ajusta con todo
    el train, como en la evaluacion final.
    """
    partes = []
    for df, prefijo, nombre in ((train, "tr", "TRAIN"), (test, "te", "TEST")):
        parte = df.copy()
        parte.insert(0, ID, [f"{prefijo}{i:05d}" for i in range(len(df))])
        parte[PARTICION] = nombre
        partes.append(parte)
    return pd.concat(partes, ignore_index=True)


def verificar_tabla_pipeline(tabla: pd.DataFrame, objetivo_origen: str,
                             excluidas: list[str]) -> list[str]:
    """Mismo contrato que la tabla de AutoML: mismo test y sin fuga."""
    problemas = []
    con_fuga = sorted(set(excluidas) & set(tabla.columns))
    if con_fuga:
        problemas.append(f"columnas con fuga: {con_fuga}")
    if not set(tabla[PARTICION]) <= {"TRAIN", "TEST"}:
        problemas.append(f"valores de particion no validos: {sorted(set(tabla[PARTICION]))}")
    if tabla[ID].duplicated().any():
        problemas.append("hay identificadores repetidos")
    test = tabla[tabla[PARTICION] == "TEST"]
    moras = int((1 - test[objetivo_origen]).sum())
    if len(test) != TEST_ESPERADO or moras != MORAS_TEST_ESPERADAS:
        problemas.append(f"el test no es el de Python: {len(test)} creditos y {moras} moras; "
                         f"se esperaban {TEST_ESPERADO} y {MORAS_TEST_ESPERADAS}")
    if (tabla[PARTICION] == "TRAIN").sum() != TRAIN_ESPERADO:
        problemas.append("el train no es el de Python")
    return problemas


def resumen(tabla: pd.DataFrame) -> pd.DataFrame:
    grupos = tabla.groupby(PARTICION, sort=False)[OBJETIVO]
    return pd.DataFrame({"creditos": grupos.size(), "moras": grupos.sum(),
                         "tasa_mora": grupos.mean().round(4)})


def subir(tabla: pd.DataFrame, destino: dict, nombre_tabla: str) -> str:
    """Reemplaza la tabla en BigQuery. La tabla es derivada y se puede rehacer."""
    import pandas_gbq

    nombre = f"{destino['dataset']}.{nombre_tabla}"
    pandas_gbq.to_gbq(tabla, nombre, project_id=destino["proyecto"],
                      if_exists="replace", location=destino["region"])
    return f"{destino['proyecto']}.{nombre}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--tabla", choices=("comparacion", "pipeline"), default="comparacion",
                        help="comparacion: 12 entradas para AutoML; pipeline: particiones "
                             "procesadas para reentrenar el modelo de Python")
    parser.add_argument("--subir", action="store_true",
                        help="reemplaza la tabla en BigQuery si pasa la verificacion")
    args = parser.parse_args(argv)

    config = cargar_config()
    train = leer_particion("estratificado_train", config)
    test = leer_particion("estratificado_test", config)
    objetivo_origen = config["target_variable"]

    if args.tabla == "pipeline":
        tabla = construir_tabla_pipeline(train, test)
        nombre_tabla = config["vertex"]["tabla_pipeline"]
        print(f"Columnas ({tabla.shape[1]}): {', '.join(tabla.columns)}")
        print(resumen(tabla.assign(**{OBJETIVO: 1 - tabla[objetivo_origen]})).to_string())
        problemas = verificar_tabla_pipeline(tabla, objetivo_origen, variables_excluidas())
    else:
        entradas = entradas_del_modelo()
        tabla = construir_tabla(
            train, test, entradas,
            objetivo_origen=objetivo_origen,
            semilla=config["random_state"],
            proporcion_validacion=config["vertex"]["proporcion_validacion"],
        )
        nombre_tabla = config["vertex"]["tabla"]
        print(f"Entradas ({len(entradas)}): {', '.join(entradas)}")
        print(resumen(tabla).to_string())
        problemas = verificar_tabla(tabla, entradas)
    for problema in problemas:
        print(f"[FALLA] {problema}")
    if problemas:
        print("RECHAZADO: la tabla no se sube.")
        return 1
    print("APROBADO: la tabla cumple las reglas de la comparacion.")

    if args.subir:
        print(f"Subida: {subir(tabla, config['vertex'], nombre_tabla)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
