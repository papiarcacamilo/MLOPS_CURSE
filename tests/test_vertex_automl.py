"""
Pruebas del lanzamiento de AutoML
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

AutoML cuesta cerca de USD 21 por corrida y el saldo alcanza para una. Un error
en el plan, como una columna de mas o un objetivo distinto, se pagaria entero y
dejaria la comparacion sin valor. Estas pruebas revisan el plan antes de gastar.

Ninguna prueba llama a Google Cloud: Vertex AI se reemplaza por un modulo falso.
================================================================================
"""

from __future__ import annotations

import copy
import sys
import types

import google.cloud
import pytest

import vertex_automl as va
import vertex_datos as vd


@pytest.fixture
def plan_real():
    return va.plan(vd.cargar_config(), vd.entradas_del_modelo())


def _entradas(p: dict) -> dict:
    return p["entrenamiento"]["training_task_inputs"]


def test_las_categorias_codificadas_como_numero_se_declaran_categoricas():
    lista = va.transformaciones(["tipo_credito", "capital_prestado", "tendencia_ingresos"])
    assert lista[0] == {"categorical": {"columnName": "tipo_credito"}}
    assert lista[2] == {"categorical": {"columnName": "tendencia_ingresos"}}


def test_las_numericas_conservan_la_fila_con_valores_invalidos():
    lista = va.transformaciones(["puntaje_datacredito"])
    assert lista == [{"numeric": {"columnName": "puntaje_datacredito",
                                  "invalidValuesAllowed": True}}]


def test_el_plan_real_cumple_las_reglas(plan_real):
    assert va.verificar_plan(plan_real, vd.entradas_del_modelo()) == []
    assert plan_real["origen"] == "bq://nb-rsk-scoring-dev.modelado.creditos_comparacion"
    assert plan_real["entrenamiento"]["input_data_config"] == {
        "predefined_split": {"key": "particion"}}
    assert _entradas(plan_real)["exportEvaluatedDataItemsConfig"] == {
        "destinationBigqueryUri": "bq://nb-rsk-scoring-dev:modelado:automl_predicciones_test",
        "overrideExistingTable": False}


@pytest.mark.parametrize("columna", [vd.ID, vd.OBJETIVO, vd.PARTICION, vd.COLUMNA_FECHA])
def test_una_columna_prohibida_como_entrada_se_rechaza(plan_real, columna):
    entradas = [*vd.entradas_del_modelo(), columna]
    _entradas(plan_real)["transformations"] = va.transformaciones(entradas)
    assert f"{columna} no puede ser una entrada" in va.verificar_plan(plan_real, entradas)


def test_entradas_distintas_a_las_del_modelo_se_rechazan(plan_real):
    _entradas(plan_real)["transformations"].pop()
    problemas = va.verificar_plan(plan_real, vd.entradas_del_modelo())
    assert "las entradas no son las del modelo servido" in problemas


@pytest.mark.parametrize("campo, valor, mensaje", [
    ("optimizationObjective", "maximize-au-roc", "no es AUC-PR"),
    ("trainBudgetMilliNodeHours", 2000, "no es de 1 nodo-hora"),
])
def test_objetivo_o_presupuesto_distintos_se_rechazan(plan_real, campo, valor, mensaje):
    _entradas(plan_real)[campo] = valor
    problemas = va.verificar_plan(plan_real, vd.entradas_del_modelo())
    assert any(mensaje in p for p in problemas)


def test_main_sin_lanzar_no_crea_nada(monkeypatch, capsys):
    llamadas = []
    monkeypatch.setattr(va, "lanzar", lambda p: llamadas.append(p))
    assert va.main([]) == 0
    assert llamadas == []
    assert "Sin --lanzar no se crea nada." in capsys.readouterr().out


def test_main_con_lanzar_crea_el_entrenamiento(monkeypatch, capsys):
    monkeypatch.setattr(va, "lanzar", lambda p: ("datasets/1", "trainingPipelines/2"))
    assert va.main(["--lanzar"]) == 0
    assert "Entrenamiento creado: trainingPipelines/2" in capsys.readouterr().out


def test_main_no_lanza_un_plan_rechazado(monkeypatch, capsys):
    original = va.plan

    def plan_caro(config, entradas):
        p = copy.deepcopy(original(config, entradas))
        _entradas(p)["trainBudgetMilliNodeHours"] = 5000
        return p

    llamadas = []
    monkeypatch.setattr(va, "plan", plan_caro)
    monkeypatch.setattr(va, "lanzar", lambda p: llamadas.append(p))
    assert va.main(["--lanzar"]) == 1
    assert llamadas == []
    assert "RECHAZADO" in capsys.readouterr().out


@pytest.fixture
def vertex_falso(monkeypatch):
    """Reemplaza Vertex AI y registra cada llamada."""
    registro = {"existentes": []}

    class Dataset:
        name = "987"
        resource_name = "projects/p/locations/us-central1/datasets/987"

        @staticmethod
        def list(filter):
            registro["filtro"] = filter
            return registro["existentes"]

        @staticmethod
        def create(**opciones):
            registro["dataset"] = opciones
            return Dataset()

    class Cliente:
        def __init__(self, client_options):
            registro["endpoint"] = client_options["api_endpoint"]

        def create_training_pipeline(self, parent, training_pipeline):
            registro["parent"] = parent
            registro["pipeline"] = training_pipeline
            return types.SimpleNamespace(name="trainingPipelines/555")

    falso = types.SimpleNamespace(
        init=lambda **opciones: registro.setdefault("init", opciones),
        TabularDataset=Dataset,
        gapic=types.SimpleNamespace(PipelineServiceClient=Cliente))
    monkeypatch.setattr(google.cloud, "aiplatform", falso, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", falso)
    registro["clase_dataset"] = Dataset
    return registro


def test_preparar_dataset_reutiliza_el_existente(vertex_falso, plan_real):
    existente = types.SimpleNamespace(name="111")
    vertex_falso["existentes"].append(existente)
    assert va.preparar_dataset(plan_real) is existente
    assert vertex_falso["filtro"] == 'display_name="creditos_comparacion-e5"'
    assert "dataset" not in vertex_falso


def test_lanzar_crea_el_dataset_y_el_entrenamiento_con_el_plan(vertex_falso, plan_real):
    registro = vertex_falso
    Dataset = registro["clase_dataset"]

    assert va.lanzar(plan_real) == (Dataset.resource_name, "trainingPipelines/555")
    assert registro["init"] == {"project": "nb-rsk-scoring-dev", "location": "us-central1"}
    assert registro["dataset"]["bq_source"] == plan_real["origen"]
    assert registro["endpoint"] == "us-central1-aiplatform.googleapis.com"
    assert registro["parent"] == "projects/nb-rsk-scoring-dev/locations/us-central1"
    enviado = registro["pipeline"]
    assert enviado["input_data_config"] == {"predefined_split": {"key": "particion"},
                                           "dataset_id": "987"}
    objetivo = enviado["training_task_inputs"].struct_value["optimizationObjective"]
    assert objetivo == "maximize-au-prc"
