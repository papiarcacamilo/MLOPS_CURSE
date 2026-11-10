"""
Entregable 5, Fase 2 - Entrenamiento con AutoML Tabular en Vertex AI
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : la tabla de BigQuery que arma vertex_datos.py
Salida   : un modelo de AutoML y sus predicciones sobre el test, en BigQuery

AutoML cuesta dinero, asi que el script por defecto solo muestra lo que
lanzaria. Para lanzarlo hay que pedirlo con --lanzar.

Decisiones de la comparacion:

  - Objetivo maximize-au-prc, la misma metrica con la que se eligio el modelo
    de Python.
  - Particion manual: AutoML usa exactamente TRAIN, VALIDATE y TEST de la tabla.
  - Tipos explicitos. `tipo_credito` es un codigo numerico, pero sus valores
    son categorias: tratarlo como numero diria que 7 esta entre 6 y 9.
  - Las numericas admiten valores invalidos. Si no, AutoML descarta la fila, y
    con los nulos de `promedio_ingresos_datacredito` el test perderia creditos.
  - `id_fila` no es una entrada: solo sirve para rastrear cada prediccion.
  - Las predicciones sobre el test se exportan a BigQuery para medirlas con la
    misma funcion que el modelo de Python, sin endpoint ni prediccion por lote.

El entrenamiento corre en Vertex AI. El script lo crea y termina: cerrar la
terminal no lo detiene.

Uso:
    python mlops_pipeline/src/vertex_automl.py              muestra el plan
    python mlops_pipeline/src/vertex_automl.py --lanzar     crea el entrenamiento
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import vertex_datos as vd

DEFINICION = ("gs://google-cloud-aiplatform/schema/trainingjob/definition/"
              "automl_tabular_1.0.0.yaml")
CATEGORICAS = ("tipo_credito", "tendencia_ingresos")
ETIQUETAS = {"entregable": "e5", "dominio": "riesgo", "uso": "comparacion-modelos"}


def transformaciones(entradas: list[str]) -> list[dict]:
    """Tipo explicito para cada entrada. Lo que no esta aqui no entra al modelo."""
    lista = []
    for columna in entradas:
        if columna in CATEGORICAS:
            lista.append({"categorical": {"columnName": columna}})
        else:
            lista.append({"numeric": {"columnName": columna,
                                      "invalidValuesAllowed": True}})
    return lista


def plan(config: dict, entradas: list[str]) -> dict:
    """Todo lo que se va a crear, en un solo lugar, para revisarlo antes."""
    v, a = config["vertex"], config["vertex"]["automl"]
    return {
        "proyecto": v["proyecto"],
        "region": v["region"],
        "origen": f"bq://{v['proyecto']}.{v['dataset']}.{v['tabla']}",
        "dataset_vertex": f"{v['tabla']}-e5",
        "entrenamiento": {
            "display_name": a["modelo"],
            "training_task_definition": DEFINICION,
            "training_task_inputs": {
                "predictionType": "classification",
                "targetColumn": vd.OBJETIVO,
                "optimizationObjective": a["objetivo"],
                "transformations": transformaciones(entradas),
                "trainBudgetMilliNodeHours": a["presupuesto_milli_nodo_horas"],
                "disableEarlyStopping": False,
                "exportEvaluatedDataItemsConfig": {
                    "destinationBigqueryUri": (f"bq://{v['proyecto']}:{v['dataset']}:"
                                               f"{a['tabla_predicciones']}"),
                    "overrideExistingTable": False,
                },
            },
            "input_data_config": {"predefined_split": {"key": vd.PARTICION}},
            "model_to_upload": {"display_name": a["modelo"], "labels": ETIQUETAS},
            "labels": ETIQUETAS,
        },
    }


def verificar_plan(p: dict, entradas: list[str]) -> list[str]:
    """Reglas que el plan debe cumplir antes de gastar el presupuesto."""
    problemas = []
    entradas_plan = p["entrenamiento"]["training_task_inputs"]
    columnas = [list(t.values())[0]["columnName"] for t in entradas_plan["transformations"]]
    if columnas != entradas:
        problemas.append("las entradas no son las del modelo servido")
    for prohibida in (vd.ID, vd.OBJETIVO, vd.PARTICION, vd.COLUMNA_FECHA):
        if prohibida in columnas:
            problemas.append(f"{prohibida} no puede ser una entrada")
    if entradas_plan["optimizationObjective"] != "maximize-au-prc":
        problemas.append("el objetivo no es AUC-PR, el del modelo de Python")
    if entradas_plan["trainBudgetMilliNodeHours"] != 1000:
        problemas.append("el presupuesto no es de 1 nodo-hora; el saldo solo alcanza para uno")
    return problemas


def preparar_dataset(p: dict):
    """Devuelve el dataset de Vertex AI; lo crea solo si no existe, para no duplicarlo."""
    from google.cloud import aiplatform

    aiplatform.init(project=p["proyecto"], location=p["region"])
    existentes = aiplatform.TabularDataset.list(
        filter=f'display_name="{p["dataset_vertex"]}"')
    if existentes:
        return existentes[0]
    return aiplatform.TabularDataset.create(
        display_name=p["dataset_vertex"], bq_source=p["origen"], labels=ETIQUETAS)


def lanzar(p: dict) -> tuple[str, str]:
    """Crea el entrenamiento sin esperar a que termine."""
    from google.cloud import aiplatform
    from google.protobuf import json_format, struct_pb2

    dataset = preparar_dataset(p)

    entrenamiento = dict(p["entrenamiento"])
    entrenamiento["training_task_inputs"] = json_format.ParseDict(
        entrenamiento["training_task_inputs"], struct_pb2.Value())
    entrenamiento["input_data_config"] = {**entrenamiento["input_data_config"],
                                          "dataset_id": dataset.name}

    cliente = aiplatform.gapic.PipelineServiceClient(
        client_options={"api_endpoint": f"{p['region']}-aiplatform.googleapis.com"})
    creado = cliente.create_training_pipeline(
        parent=f"projects/{p['proyecto']}/locations/{p['region']}",
        training_pipeline=entrenamiento)
    return dataset.resource_name, creado.name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--lanzar", action="store_true",
                        help="crea el entrenamiento; sin esto solo muestra el plan")
    args = parser.parse_args(argv)

    config = vd.cargar_config()
    entradas = vd.entradas_del_modelo()
    p = plan(config, entradas)
    print(json.dumps(p, indent=2, ensure_ascii=False))

    problemas = verificar_plan(p, entradas)
    for problema in problemas:
        print(f"[FALLA] {problema}")
    if problemas:
        print("RECHAZADO: el plan no se lanza.")
        return 1
    print("APROBADO: el plan cumple las reglas de la comparacion.")

    if not args.lanzar:
        print("Sin --lanzar no se crea nada.")
        return 0
    dataset, entrenamiento = lanzar(p)
    print(f"Dataset de Vertex AI: {dataset}")
    print(f"Entrenamiento creado: {entrenamiento}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
