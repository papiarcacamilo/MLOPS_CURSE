"""
Pruebas del pipeline propio en Vertex AI Pipelines
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

El pipeline vale si reentrena el mismo modelo que se publico y si solo se lanza
desde un commit rastreable. La prueba central corre las etapas de verdad, en
local, y exige el AUC-PR publicado de 0,1479.

Ninguna prueba toca Google Cloud: el almacenamiento, BigQuery y Vertex AI se
reemplazan por objetos falsos, y git tambien.
================================================================================
"""

from __future__ import annotations

import io
import json
import sys
import types
import zipfile

import google.cloud
import pandas as pd
import pytest
import yaml

import vertex_datos as vd
import vertex_evaluacion as ve
import vertex_pasos as vp
import vertex_pipeline as vpi

CONFIG = vd.cargar_config()
TABLA = "nb-rsk-scoring-dev.modelado.creditos_pipeline"


def _tabla_pipeline() -> pd.DataFrame:
    return vd.construir_tabla_pipeline(vd.leer_particion("estratificado_train", CONFIG),
                                       vd.leer_particion("estratificado_test", CONFIG))


def _cliente_bigquery(tabla: pd.DataFrame, consultas: list | None = None):
    def query(sql, location):
        if consultas is not None:
            consultas.append((sql, location))
        return types.SimpleNamespace(to_dataframe=lambda: tabla.copy())
    return types.SimpleNamespace(query=query)


def _artefacto(ruta, uri="gs://falso/artefacto"):
    return types.SimpleNamespace(path=str(ruta), uri=uri, metadata={})


@pytest.fixture(scope="module")
def particiones(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("datos")
    conteos = vp.cargar_datos(TABLA, "us-central1", str(carpeta / "train.csv"),
                              str(carpeta / "test.csv"), cliente=_cliente_bigquery(_tabla_pipeline()))
    return carpeta, conteos


# --- Etapas ----------------------------------------------------------------------


def test_cargar_datos_separa_train_y_test_sin_columnas_de_mas(particiones):
    carpeta, conteos = particiones
    assert conteos == {"train": 8610, "test": 2153, "moras_test": 102}
    train = pd.read_csv(carpeta / "train.csv", sep=";")
    test = pd.read_csv(carpeta / "test.csv", sep=";")
    assert vd.ID not in train.columns and vd.PARTICION not in train.columns
    assert train.equals(vd.leer_particion("estratificado_train", CONFIG))
    assert test.drop(columns=[vd.ID]).equals(vd.leer_particion("estratificado_test", CONFIG))


def test_cargar_datos_ordena_la_consulta_por_id(tmp_path):
    consultas = []
    vp.cargar_datos(TABLA, "us-central1", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
                    cliente=_cliente_bigquery(_tabla_pipeline(), consultas))
    assert consultas == [(f"SELECT * FROM `{TABLA}` ORDER BY id_fila", "us-central1")]


def test_cargar_datos_rechaza_filas_fuera_de_train_y_test(tmp_path):
    tabla = _tabla_pipeline()
    tabla.loc[0, vd.PARTICION] = "VALIDATE"
    with pytest.raises(ValueError, match="fuera de TRAIN y TEST"):
        vp.cargar_datos(TABLA, "us-central1", str(tmp_path / "a.csv"),
                        str(tmp_path / "b.csv"), cliente=_cliente_bigquery(tabla))


def test_entrenar_y_evaluar_reproducen_el_modelo_publicado(particiones, tmp_path):
    carpeta, _ = particiones
    info = vp.entrenar(str(carpeta / "train.csv"), str(tmp_path / "modelo"))
    assert info == {"creditos": 8610, "moras": 409}
    m = vp.evaluar(str(tmp_path / "modelo"), str(carpeta / "test.csv"), str(tmp_path / "p.csv"))
    assert m["auc_pr"] == 0.1479
    predicciones = pd.read_csv(tmp_path / "p.csv")
    referencia = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV)
    assert predicciones[vd.ID].equals(referencia[vd.ID])
    assert (predicciones["p_pipeline"] - referencia["p_python"]).abs().max() < 1e-12


def test_registrar_sube_el_modelo_con_sus_etiquetas():
    llamadas = {}

    class Model:
        @staticmethod
        def upload(**opciones):
            llamadas["upload"] = opciones
            return types.SimpleNamespace(resource_name="models/7")

    falso = types.SimpleNamespace(init=lambda **o: llamadas.setdefault("init", o), Model=Model)
    nombre = vp.registrar("gs://b/modelo", "p", "r", "python-woe", {"commit": "abc"}, falso)
    assert nombre == "models/7"
    assert llamadas["init"] == {"project": "p", "location": "r"}
    assert llamadas["upload"]["artifact_uri"] == "gs://b/modelo"
    assert llamadas["upload"]["labels"] == {"commit": "abc"}


# --- Componentes de Kubeflow ----------------------------------------------------


def _zip_codigo(faltante: str | None = None) -> bytes:
    contenido = io.BytesIO()
    with zipfile.ZipFile(contenido, "w") as z:
        for archivo in ("config.json", "Base_de_datos.csv", "mlops_pipeline/src/vertex_pasos.py",
                        "data/models/modelo_seleccionado.joblib"):
            if archivo != faltante:
                z.writestr(archivo, "x")
    return contenido.getvalue()


def _storage_falso(monkeypatch, datos: bytes, pedidos: list):
    from google.cloud import storage

    class Cliente:
        def bucket(self, nombre):
            def blob(ruta):
                pedidos.append((nombre, ruta))
                return types.SimpleNamespace(download_as_bytes=lambda: datos)
            return types.SimpleNamespace(blob=blob)

    monkeypatch.setattr(storage, "Client", Cliente)


def test_preparar_codigo_extrae_el_paquete_del_commit(monkeypatch, tmp_path):
    pedidos = []
    _storage_falso(monkeypatch, _zip_codigo(), pedidos)
    codigo = _artefacto(tmp_path / "codigo")
    vpi.preparar_codigo.python_func(codigo_uri="gs://bucket/codigo/abc.zip", codigo=codigo)
    assert pedidos == [("bucket", "codigo/abc.zip")]
    assert (tmp_path / "codigo" / "mlops_pipeline" / "src" / "vertex_pasos.py").exists()
    assert codigo.metadata["origen"] == "gs://bucket/codigo/abc.zip"


def test_preparar_codigo_falla_si_falta_la_receta(monkeypatch, tmp_path):
    _storage_falso(monkeypatch, _zip_codigo("data/models/modelo_seleccionado.joblib"), [])
    with pytest.raises(FileNotFoundError, match="modelo_seleccionado"):
        vpi.preparar_codigo.python_func(codigo_uri="gs://b/c.zip",
                                        codigo=_artefacto(tmp_path / "codigo"))


def test_los_componentes_conectan_las_etapas_en_local(monkeypatch, tmp_path):
    from google.cloud import bigquery

    monkeypatch.setattr(bigquery, "Client",
                        lambda project: _cliente_bigquery(_tabla_pipeline()))
    subidas = {}

    class Model:
        @staticmethod
        def upload(**opciones):
            subidas.update(opciones)
            return types.SimpleNamespace(resource_name="models/9")

    falso = types.SimpleNamespace(init=lambda **o: None, Model=Model)
    monkeypatch.setattr(google.cloud, "aiplatform", falso, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", falso)

    codigo = _artefacto(vd.RAIZ)
    train, test = _artefacto(tmp_path / "train.csv"), _artefacto(tmp_path / "test.csv")
    vpi.cargar_datos.python_func(codigo=codigo, tabla=TABLA, region="us-central1",
                                 train=train, test=test)
    assert test.metadata == {"train": 8610, "test": 2153, "moras_test": 102}

    modelo = _artefacto(tmp_path / "modelo", uri="gs://b/modelo")
    vpi.entrenar.python_func(codigo=codigo, train=train, modelo=modelo)
    assert modelo.metadata["framework"] == "scikit-learn"

    registradas = {}
    metricas = types.SimpleNamespace(log_metric=lambda k, v: registradas.update({k: v}))
    salida = vpi.evaluar.python_func(codigo=codigo, modelo=modelo, test=test, metricas=metricas,
                                     predicciones=_artefacto(tmp_path / "p.csv"))
    assert salida.auc_pr == 0.1479 and registradas["auc_pr"] == 0.1479
    assert set(registradas) >= {"roc_auc", "ks", "brier", "f1", "accuracy"}

    nombre = vpi.registrar.python_func(codigo=codigo, modelo=modelo, proyecto="p", region="r",
                                       nombre="python-woe", commit="0123456789abcdef")
    assert nombre == "models/9"
    assert subidas["labels"]["commit"] == "0123456789ab"


# --- Compilacion -----------------------------------------------------------------


def test_las_versiones_salen_de_requirements():
    assert vpi.fijadas(["pandas", "scikit-learn"]) == ["pandas==3.0.5", "scikit-learn==1.9.0"]
    assert vpi.fijadas(["google-cloud-aiplatform"]) == ["google-cloud-aiplatform==2.7.0"]
    with pytest.raises(ValueError, match="sin version fijada"):
        vpi.fijadas(["paquete-inexistente"])


def test_el_minimo_para_registrar_es_el_de_jenkins():
    from validar_metricas import AUC_PR_MINIMO
    assert vpi.auc_pr_minimo() == AUC_PR_MINIMO == 0.132


def test_el_pipeline_compila_con_sus_etapas_y_la_condicion(tmp_path):
    spec = yaml.safe_load(vpi.compilar(tmp_path / "p.yaml").read_text(encoding="utf-8"))
    tareas = spec["root"]["dag"]["tasks"]
    assert set(tareas) == {"preparar-codigo", "cargar-datos", "entrenar", "evaluar", "condition-1"}
    condicion = tareas["condition-1"]["triggerPolicy"]["condition"]
    assert condicion == "inputs.parameter_values['pipelinechannel--evaluar-auc_pr'] >= 0.132"
    assert list(spec["components"]["comp-condition-1"]["dag"]["tasks"]) == ["registrar"]
    comando = " ".join(spec["deploymentSpec"]["executors"]["exec-entrenar"]["container"]["command"])
    assert all(p in comando for p in vpi.PAQUETES)


def test_el_yaml_versionado_esta_al_dia(tmp_path):
    nuevo = vpi.compilar(tmp_path / "p.yaml").read_text(encoding="utf-8")
    assert nuevo == vpi.RUTA_YAML.read_text(encoding="utf-8"), \
        "recompila con: python mlops_pipeline/src/vertex_pipeline.py"


# --- Lanzamiento -----------------------------------------------------------------


def _git_falso(estado="", ramas="origin/feature1", commit="f" * 40, llamadas=None):
    respuestas = {"status": estado, "rev-parse": commit, "branch": ramas, "archive": ""}

    def git(*args):
        if llamadas is not None:
            llamadas.append(args)
        return respuestas[args[0]]
    return git


def test_no_se_lanza_con_cambios_sin_commit():
    with pytest.raises(RuntimeError, match="cambios sin commit"):
        vpi.commit_lanzable(_git_falso(estado=" M config.json"))


def test_no_se_lanza_un_commit_que_no_esta_en_github():
    with pytest.raises(RuntimeError, match="no esta en GitHub"):
        vpi.commit_lanzable(_git_falso(ramas=""))


def test_un_commit_limpio_y_publicado_se_puede_lanzar():
    assert vpi.commit_lanzable(_git_falso()) == "f" * 40


def test_el_paquete_solo_lleva_lo_que_el_pipeline_necesita(tmp_path):
    llamadas = []
    vpi.empaquetar("abc", tmp_path / "abc.zip", _git_falso(llamadas=llamadas))
    assert llamadas == [("archive", "--format=zip", "-o", str(tmp_path / "abc.zip"), "abc", "--",
                         *vpi.ARCHIVOS_CODIGO)]
    assert "mlops_pipeline/src/*.py" in vpi.ARCHIVOS_CODIGO


@pytest.mark.parametrize("existe, subidas_esperadas", [(False, 1), (True, 0)])
def test_el_codigo_se_sube_una_vez_por_commit(tmp_path, existe, subidas_esperadas):
    subidas = []
    blob = types.SimpleNamespace(exists=lambda: existe,
                                 upload_from_filename=lambda ruta: subidas.append(ruta))
    cliente = types.SimpleNamespace(bucket=lambda nombre: types.SimpleNamespace(blob=lambda r: blob))
    uri = vpi.subir_codigo(tmp_path / "abc.zip", "bucket", "abc", cliente)
    assert uri == "gs://bucket/codigo/abc.zip"
    assert len(subidas) == subidas_esperadas


def test_el_plan_usa_la_tabla_la_cuenta_y_las_etiquetas_configuradas():
    p = vpi.plan(CONFIG, "0123456789abcdef", "gs://b/c.zip")
    assert p["parametros"]["tabla"] == TABLA
    assert p["cuenta_servicio"] == "sa-entrenamiento@nb-rsk-scoring-dev.iam.gserviceaccount.com"
    assert p["pipeline_root"] == "gs://nb-rsk-scoring-dev-vertex/pipelines"
    assert p["etiquetas"]["commit"] == "0123456789ab"
    assert p["display_name"] == "mlops-mora-pipeline-e5-0123456"


def test_lanzar_crea_la_corrida_sin_cache_con_la_cuenta_de_servicio(tmp_path):
    registro = {}

    class PipelineJob:
        resource_name = "pipelineJobs/1"

        def __init__(self, **opciones):
            registro["job"] = opciones

        def submit(self, service_account):
            registro["cuenta"] = service_account

    falso = types.SimpleNamespace(init=lambda **o: registro.setdefault("init", o),
                                  PipelineJob=PipelineJob)
    p = vpi.plan(CONFIG, "0123456789abcdef", "gs://b/c.zip")
    assert vpi.lanzar(p, tmp_path / "p.yaml", falso) == "pipelineJobs/1"
    assert registro["job"]["enable_caching"] is False
    assert registro["job"]["parameter_values"] == p["parametros"]
    assert registro["cuenta"] == p["cuenta_servicio"]


def test_main_sin_lanzar_solo_compila(tmp_path, capsys):
    assert vpi.main(["--salida", str(tmp_path / "p.yaml")]) == 0
    assert (tmp_path / "p.yaml").exists()
    assert "Sin --lanzar no se sube ni se crea nada." in capsys.readouterr().out


def test_main_con_lanzar_sube_el_codigo_y_crea_la_corrida(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(vpi, "commit_lanzable", lambda: "0123456789abcdef")
    monkeypatch.setattr(vpi, "empaquetar", lambda commit, destino: destino)
    monkeypatch.setattr(vpi, "subir_codigo", lambda paquete, bucket, commit: f"gs://{bucket}/c.zip")
    monkeypatch.setattr(vpi, "lanzar", lambda p, plantilla: "pipelineJobs/5")
    assert vpi.main(["--lanzar", "--salida", str(tmp_path / "p.yaml")]) == 0
    salida = capsys.readouterr().out
    assert "Corrida creada: pipelineJobs/5" in salida
    assert json.loads(salida[salida.index("{"):salida.rindex("}") + 1])["parametros"][
        "codigo_uri"] == "gs://nb-rsk-scoring-dev-vertex/c.zip"


@pytest.mark.skipif(__import__("shutil").which("git") is None,
                    reason="el agente de Jenkins corre en una imagen sin git")
def test_git_corre_en_la_raiz_del_repositorio():
    assert vpi._git("rev-parse", "--show-toplevel").lower().replace("\\", "/").endswith(
        vd.RAIZ.name.lower())


def test_subir_codigo_usa_el_cliente_de_storage_por_defecto(monkeypatch, tmp_path):
    from google.cloud import storage
    blob = types.SimpleNamespace(exists=lambda: True, upload_from_filename=lambda r: None)
    monkeypatch.setattr(storage, "Client",
                        lambda: types.SimpleNamespace(bucket=lambda n: types.SimpleNamespace(
                            blob=lambda r: blob)))
    assert vpi.subir_codigo(tmp_path / "a.zip", "b", "c") == "gs://b/codigo/c.zip"


def test_lanzar_usa_vertex_ai_por_defecto(monkeypatch, tmp_path):
    class PipelineJob:
        resource_name = "pipelineJobs/3"

        def __init__(self, **opciones):
            pass

        def submit(self, service_account):
            pass

    falso = types.SimpleNamespace(init=lambda **o: None, PipelineJob=PipelineJob)
    monkeypatch.setattr(google.cloud, "aiplatform", falso, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", falso)
    p = vpi.plan(CONFIG, "0123456789abcdef", "gs://b/c.zip")
    assert vpi.lanzar(p, tmp_path / "p.yaml") == "pipelineJobs/3"
