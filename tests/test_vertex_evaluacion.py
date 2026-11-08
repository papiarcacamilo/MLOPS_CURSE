"""
Pruebas de la comparacion entre AutoML y el modelo de Python
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

La comparacion solo vale si las dos probabilidades se miden igual y sobre los
mismos creditos. Estas pruebas cuidan el emparejamiento, el volumen de rechazo
y el veredicto contra el ruido entre pliegues.

Ninguna prueba consulta BigQuery ni escribe en data/models.
================================================================================
"""

from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

import vertex_datos as vd
import vertex_evaluacion as ve

ENTRADAS = ["capital_prestado", "tendencia_ingresos"]


def _test_local() -> pd.DataFrame:
    return pd.DataFrame({
        "capital_prestado": [100, 200, 300, 400],
        "tendencia_ingresos": ["Estable", "Creciente", "Estable", "Sin_dato"],
        vd.OBJETIVO: [0, 1, 0, 0],
    })


def _export(orden=(3, 1, 0, 2)) -> pd.DataFrame:
    # BigQuery exporta los enteros como texto y en cualquier orden.
    local = _test_local()
    export = local.iloc[list(orden)].astype(str)
    export["p_automl"] = [0.4, 0.2, 0.1, 0.3]
    return export


def test_alinear_devuelve_la_probabilidad_en_el_orden_del_test():
    p = ve.alinear(_test_local(), _export(), ENTRADAS)
    assert list(p) == [0.1, 0.2, 0.3, 0.4]


def test_alinear_rechaza_un_credito_sin_prediccion():
    export = _export().iloc[:3]
    with pytest.raises(ValueError, match="1 creditos del test no tienen prediccion"):
        ve.alinear(_test_local(), export, ENTRADAS)


def test_alinear_rechaza_creditos_repetidos():
    local = pd.concat([_test_local(), _test_local().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repetidos"):
        ve.alinear(local, _export(), ENTRADAS)


def test_medir_rechaza_el_volumen_pedido_y_cuadra_la_matriz():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, 2000)
    p = np.clip(y * 0.1 + rng.random(2000) * 0.3, 0, 1)
    m = ve.medir(y, p, k=0.209)
    assert m["pct_marcado"] == pytest.approx(20.9, abs=0.1)
    matriz = m["matriz"]
    assert sum(matriz.values()) == 2000
    assert m["accuracy"] == round((matriz["tp"] + matriz["tn"]) / 2000, 4)
    assert m["rmse"] == pytest.approx(np.sqrt(m["brier"]), abs=1e-3)
    assert m["accuracy_trivial"] == round(1 - y.mean(), 4)


@pytest.mark.parametrize("auc_automl, veredicto", [
    (0.1550, "empate"),
    (0.1952, "gana AutoML"),
    (0.1200, "gana Python"),
])
def test_el_veredicto_respeta_el_ruido_entre_pliegues(auc_automl, veredicto):
    base = {m: 0.0 for m in ("roc_auc", "ks", "gini", "brier", "rmse",
                             "precision", "recall", "f1", "accuracy")}
    resultado = ve.comparar({**base, "auc_pr": 0.1479}, {**base, "auc_pr": auc_automl})
    assert resultado["veredicto_auc_pr"] == veredicto


def test_python_debe_reproducir_su_auc_pr_publicado(tmp_path):
    (tmp_path / "evaluacion.json").write_text(json.dumps(
        {"evaluacion_final": {"estratificado": {"metricas": {"auc_pr": 1.0}}}}))
    y = np.array([0, 1, 0, 1])
    ve.verificar_python(y, np.array([0.1, 0.9, 0.2, 0.8]), ruta=tmp_path)
    with pytest.raises(ValueError, match="no reproduce"):
        ve.verificar_python(y, np.array([0.9, 0.1, 0.8, 0.2]), ruta=tmp_path)


def test_leer_export_saca_la_probabilidad_de_mora():
    filas = pd.DataFrame({"mora": ["0"], "predicted_mora": [
        {"classes": np.array(["0", "1"]), "scores": np.array([0.97, 0.03])}]})
    consultas = []

    def query(sql, location):
        consultas.append((sql, location))
        return types.SimpleNamespace(to_dataframe=lambda: filas.copy())

    export = ve.leer_export_automl(vd.cargar_config(), types.SimpleNamespace(query=query))
    assert export["p_automl"].tolist() == [0.03]
    assert "predicted_mora" not in export.columns
    assert consultas == [("SELECT * FROM `nb-rsk-scoring-dev.modelado.automl_predicciones_test`",
                          "us-central1")]


def test_sin_cliente_usa_el_proyecto_de_riesgo(monkeypatch):
    from google.cloud import bigquery
    proyectos = []

    class Cliente:
        def __init__(self, project):
            proyectos.append(project)

        def query(self, sql, location):
            vacio = pd.DataFrame({"predicted_mora": [{"classes": ["1"], "scores": [0.2]}]})
            return types.SimpleNamespace(to_dataframe=lambda: vacio)

    monkeypatch.setattr(bigquery, "Client", Cliente)
    assert ve.leer_export_automl(vd.cargar_config())["p_automl"].tolist() == [0.2]
    assert proyectos == ["nb-rsk-scoring-dev"]


def test_las_predicciones_versionadas_reproducen_la_comparacion(tmp_path, capsys):
    assert ve.main(["--salida", str(tmp_path)]) == 0
    resultado = json.loads((tmp_path / ve.NOMBRE_JSON).read_text(encoding="utf-8"))
    assert resultado["test"] == {"creditos": 2153, "moras": 102}
    assert resultado["python"]["auc_pr"] == 0.1479
    assert resultado["python"]["pct_marcado"] == resultado["automl"]["pct_marcado"]
    assert (tmp_path / ve.NOMBRE_FIGURA).exists()
    salida = capsys.readouterr().out
    assert "Veredicto en AUC-PR" in salida and "Pipeline frente a Python" in salida


def test_el_pipeline_en_la_nube_reproduce_a_python():
    # Es la prueba de que entrenar en Vertex AI da el mismo modelo que en local.
    resultado = json.loads((ve.RUTA_MODELOS / ve.NOMBRE_JSON).read_text(encoding="utf-8"))
    assert resultado["pipeline"]["auc_pr"] == resultado["python"]["auc_pr"] == 0.1479
    assert resultado["comparacion_pipeline"]["veredicto_auc_pr"] == "empate"
    assert all(d == 0 for d in resultado["comparacion_pipeline"]["diferencias"].values())
    assert resultado["reproducibilidad"]["diferencia_maxima_de_probabilidad"] < 1e-12


def test_descargar_arma_el_csv_con_el_test_alineado(tmp_path, monkeypatch):
    guardado = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV)
    monkeypatch.setattr(ve, "leer_export_automl", lambda config: None)
    monkeypatch.setattr(ve, "leer_predicciones_pipeline", lambda config: None)
    monkeypatch.setattr(ve, "preparar_predicciones",
                        lambda config, entradas, export, pipeline: guardado)
    assert ve.main(["--descargar", "--salida", str(tmp_path)]) == 0
    assert pd.read_csv(tmp_path / ve.NOMBRE_CSV).equals(guardado)


def _corrida_falsa(estado="PIPELINE_STATE_SUCCEEDED"):
    artefacto = types.SimpleNamespace(uri="gs://bucket/ruta/predicciones")
    evaluar = types.SimpleNamespace(
        task_name="evaluar",
        outputs={"predicciones": types.SimpleNamespace(artifacts=[artefacto])})
    otra = types.SimpleNamespace(task_name="entrenar", outputs={})
    return types.SimpleNamespace(state=types.SimpleNamespace(name=estado),
                                 job_detail=types.SimpleNamespace(task_details=[otra, evaluar]))


def _almacenamiento_falso(pedidos):
    datos = b"id_fila,mora,p_pipeline\nte00001,0,0.2\nte00000,1,0.1\n"

    def bucket(nombre):
        def blob(ruta):
            pedidos.append((nombre, ruta))
            return types.SimpleNamespace(download_as_bytes=lambda: datos)
        return types.SimpleNamespace(blob=blob)
    return types.SimpleNamespace(bucket=bucket)


def test_leer_predicciones_del_pipeline_desde_su_corrida():
    corridas, pedidos = [], []
    cliente = types.SimpleNamespace(
        get_pipeline_job=lambda name: corridas.append(name) or _corrida_falsa())
    tabla = ve.leer_predicciones_pipeline(vd.cargar_config(), cliente, _almacenamiento_falso(pedidos))
    assert list(tabla.columns) == [vd.ID, "p_pipeline"]
    assert pedidos == [("bucket", "ruta/predicciones")]
    assert corridas[0].endswith("pipelineJobs/mlops-mora-pipeline-e5-20261107212012")


def test_una_corrida_que_no_termino_bien_se_rechaza():
    cliente = types.SimpleNamespace(
        get_pipeline_job=lambda name: _corrida_falsa("PIPELINE_STATE_FAILED"))
    with pytest.raises(ValueError, match="PIPELINE_STATE_FAILED"):
        ve.leer_predicciones_pipeline(vd.cargar_config(), cliente, _almacenamiento_falso([]))


def test_leer_predicciones_del_pipeline_usa_los_clientes_por_defecto(monkeypatch):
    import sys

    import google.cloud
    from google.cloud import storage
    endpoints = []

    def cliente(client_options):
        endpoints.append(client_options["api_endpoint"])
        return types.SimpleNamespace(get_pipeline_job=lambda name: _corrida_falsa())

    falso = types.SimpleNamespace(gapic=types.SimpleNamespace(PipelineServiceClient=cliente))
    monkeypatch.setattr(google.cloud, "aiplatform", falso, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", falso)
    monkeypatch.setattr(storage, "Client", lambda project: _almacenamiento_falso([]))
    assert len(ve.leer_predicciones_pipeline(vd.cargar_config())) == 2
    assert endpoints == ["us-central1-aiplatform.googleapis.com"]


def test_alinear_pipeline_ordena_por_id_y_exige_el_mismo_test():
    pipeline = pd.DataFrame({vd.ID: ["te00001", "te00000"], "p_pipeline": [0.2, 0.1]})
    ids = pd.Series(["te00000", "te00001"])
    assert list(ve.alinear_pipeline(ids, pipeline)) == [0.1, 0.2]
    with pytest.raises(ValueError, match="no son las del test"):
        ve.alinear_pipeline(pd.Series(["te00000", "te00002"]), pipeline)


def test_el_veredicto_nombra_al_modelo_comparado():
    base = {m: 0.0 for m in ("roc_auc", "ks", "gini", "brier", "rmse",
                             "precision", "recall", "f1", "accuracy")}
    resultado = ve.comparar({**base, "auc_pr": 0.10}, {**base, "auc_pr": 0.20},
                            nombre="el pipeline")
    assert resultado["veredicto_auc_pr"] == "gana el pipeline"


def test_predicciones_de_otro_test_se_rechazan(tmp_path, monkeypatch, capsys):
    corto = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV).iloc[:100]
    monkeypatch.setattr(ve.pd, "read_csv", lambda ruta: corto)
    assert ve.main(["--salida", str(tmp_path)]) == 1
    assert "RECHAZADO" in capsys.readouterr().out


def test_preparar_predicciones_une_etiqueta_y_probabilidades(monkeypatch):
    config = vd.cargar_config()
    entradas = vd.entradas_del_modelo()
    monkeypatch.setattr(ve, "alinear", lambda test, export, e: np.full(len(test), 0.5))
    guardado = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV)
    pipeline = guardado[[vd.ID, "p_pipeline"]].iloc[::-1]
    tabla = ve.preparar_predicciones(config, entradas, export=None, pipeline=pipeline)
    assert list(tabla.columns) == [vd.ID, vd.OBJETIVO, "p_python", "p_pipeline", "p_automl"]
    assert tabla["p_pipeline"].equals(guardado["p_pipeline"])
    assert len(tabla) == 2153 and tabla[vd.OBJETIVO].sum() == 102
    assert tabla[vd.ID].iloc[0] == "te00000"
