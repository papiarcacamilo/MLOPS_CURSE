"""
La API
================================================================================
Cubre app.py con el TestClient de FastAPI, que ejecuta la aplicacion en proceso:
no hace falta levantar uvicorn ni el contenedor.

EL REGISTRO DE PRODUCCION NO SE TOCA

El endpoint registra cada decision, y ese archivo es el insumo de la Fase 5. Una
suite que lo ensuciara con solicitudes de prueba falsearia la medicion de deriva:
apareceria un lote de perfiles inventados como si fueran clientes reales.

Por eso el fixture redirige `md.RUTA_REGISTRO` a un directorio temporal antes de
arrancar la app, y lo devuelve a su sitio al terminar.
================================================================================
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as api
import ft_engineering as fe
import model_deploy as md

from conftest import solicitud_base


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    """App en proceso, con el registro apuntando a un temporal."""
    temporal = tmp_path_factory.mktemp("registro")
    original_registro, original_dir = md.RUTA_REGISTRO, md.RUTA_MONITOREO
    md.RUTA_REGISTRO = temporal / "registro_endpoint.csv"
    md.RUTA_MONITOREO = temporal
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        md.RUTA_REGISTRO, md.RUTA_MONITOREO = original_registro, original_dir


def cuerpo(*solicitudes: dict) -> dict:
    return {"origen": "test", "solicitudes": list(solicitudes)}


# ==============================================================================
# LOS CINCO ENDPOINTS RESPONDEN
# ==============================================================================


def test_salud_responde_activo(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["estado"] == "activo"


def test_salud_declara_la_version_y_el_umbral(cliente):
    """La sonda del contenedor parsea esta respuesta y exige `estado`."""
    d = cliente.get("/salud").json()
    assert d["version_modelo"] == md.version_modelo()
    assert d["umbral"] == 0.0661


def test_modelo_publica_el_desempenio_medido(cliente):
    d = cliente.get("/modelo").json()
    assert d["modelo"]["n_caracteristicas"] == 17
    assert d["desempenio_en_test"]["auc_pr"] == 0.1479
    assert len(d["bandas"]) == 5


def test_modelo_publica_el_esquema_de_entrada(cliente):
    d = cliente.get("/modelo").json()
    assert d["esquema_entrada"]["columnas_requeridas"] == md.COLUMNAS_REQUERIDAS


def test_scorecard_publica_la_tabla_completa(cliente):
    d = cliente.get("/scorecard").json()
    assert len(d["tabla"]) == 45
    assert len(d["ajustes_continuos"]) == 8
    assert d["puntos_base"] == 601


def test_la_documentacion_interactiva_se_sirve(cliente):
    assert cliente.get("/docs").status_code == 200
    assert cliente.get("/openapi.json").status_code == 200


# ==============================================================================
# PREDICCION POR LOTE EN JSON
# ==============================================================================


def test_un_lote_valido_devuelve_una_decision_por_solicitud(cliente):
    r = cliente.post("/predecir", json=cuerpo(solicitud_base(),
                                              solicitud_base()))
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_la_respuesta_trae_los_cinco_campos(cliente):
    d = cliente.post("/predecir", json=cuerpo(solicitud_base())).json()[0]
    assert set(d) == {"decision", "probabilidad_mora", "puntaje", "banda",
                      "motivo"}


def test_el_orden_de_la_respuesta_es_el_de_la_peticion(cliente):
    """El llamador aparea por posicion, sin necesidad de identificador."""
    baja, alta = solicitud_base(), solicitud_base()
    alta.update(tipo_credito=6, plazo_meses=72, puntaje_datacredito=610,
                salario_cliente=1_200_000, total_otros_prestamos=80_000_000,
                tendencia_ingresos="Decreciente")
    d = cliente.post("/predecir", json=cuerpo(baja, alta)).json()
    assert d[0]["probabilidad_mora"] < d[1]["probabilidad_mora"]


def test_un_rechazo_tecnico_devuelve_null_y_no_nan(cliente):
    """JSON no admite NaN. Sin el `astype(object)`, pandas lo devolvia a NaN y
    la respuesta reventaba con un 500 al serializar.
    """
    mala = solicitud_base()
    mala["edad_cliente"] = 12
    d = cliente.post("/predecir", json=cuerpo(mala)).json()[0]
    assert d["decision"] == md.DECISION_TECNICO
    assert d["probabilidad_mora"] is None
    assert d["puntaje"] is None
    assert d["motivo"]


# ==============================================================================
# CASOS LIMITE: DE QUIEN ES LA CULPA
# ==============================================================================


def test_un_lote_vacio_es_culpa_del_cliente(cliente):
    r = cliente.post("/predecir", json={"solicitudes": []})
    assert r.status_code == 400


def test_un_campo_faltante_lo_atrapa_pydantic(cliente):
    incompleta = solicitud_base()
    del incompleta["plazo_meses"]
    assert cliente.post("/predecir", json=cuerpo(incompleta)).status_code == 422


def test_un_tipo_equivocado_lo_atrapa_pydantic(cliente):
    mala = solicitud_base()
    mala["plazo_meses"] = "treinta y seis"
    assert cliente.post("/predecir", json=cuerpo(mala)).status_code == 422


def test_un_lote_demasiado_grande_devuelve_413(cliente, monkeypatch):
    """El tope acota la memoria del contenedor: sin el, una peticion
    suficientemente grande tumba el servicio para todos los demas.
    """
    monkeypatch.setattr(api, "MAX_SOLICITUDES", 2)
    r = cliente.post("/predecir", json=cuerpo(*[solicitud_base()] * 3))
    assert r.status_code == 413


def test_los_campos_opcionales_aceptan_nulo(cliente):
    """Su ausencia es informacion: el WoE la trata como un tramo mas."""
    s = solicitud_base()
    s["puntaje_datacredito"] = None
    s["promedio_ingresos_datacredito"] = None
    r = cliente.post("/predecir", json=cuerpo(s))
    assert r.status_code == 200


# ==============================================================================
# PREDICCION POR LOTE EN CSV, QUE ES EL MODO QUE PIDE EL ENUNCIADO
# ==============================================================================


def _csv(datos: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    datos.to_csv(buffer, sep=fe.SEPARADOR, index=False)
    return buffer.getvalue().encode(fe.ENCODING)


def test_un_csv_valido_devuelve_otro_csv(cliente, lote):
    r = cliente.post("/predecir/archivo",
                     files={"archivo": ("lote.csv", _csv(lote), "text/csv")})
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]


def test_el_csv_de_salida_trae_la_entrada_y_la_decision(cliente, lote):
    """Asi el archivo se lee solo, sin aparearlo con el de entrada."""
    r = cliente.post("/predecir/archivo",
                     files={"archivo": ("lote.csv", _csv(lote), "text/csv")})
    salida = pd.read_csv(io.StringIO(r.text), sep=fe.SEPARADOR)
    assert len(salida) == len(lote)
    for columna in ("decision", "probabilidad_mora", "puntaje", "banda", "motivo"):
        assert columna in salida.columns
    assert "puntaje_datacredito" in salida.columns


def test_un_csv_con_otro_esquema_devuelve_400(cliente):
    """Antes salia como 500: como si el servicio estuviera roto en vez de la
    peticion. Es el caso que senialo danielCH26 en la revision de la PR4.
    """
    basura = _csv(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    r = cliente.post("/predecir/archivo",
                     files={"archivo": ("basura.csv", basura, "text/csv")})
    assert r.status_code == 400
    assert "Faltan columnas requeridas" in r.json()["detail"]


def test_un_csv_vacio_devuelve_400(cliente):
    r = cliente.post("/predecir/archivo",
                     files={"archivo": ("vacio.csv", b"a;b\n", "text/csv")})
    assert r.status_code == 400


def test_ningun_caso_limite_produce_un_error_500(cliente, lote):
    """La distincion que importa: 4xx es culpa de quien llama, 5xx del servicio."""
    peticiones = [
        cliente.post("/predecir", json={"solicitudes": []}),
        cliente.post("/predecir", json=cuerpo({"tipo_credito": 4})),
        cliente.post("/predecir/archivo",
                     files={"archivo": ("x.csv", b"a;b\n1;2\n", "text/csv")}),
        cliente.post("/predecir/archivo",
                     files={"archivo": ("x.csv", b"", "text/csv")}),
    ]
    assert all(r.status_code < 500 for r in peticiones), \
        [r.status_code for r in peticiones]


# ==============================================================================
# EL REGISTRO, QUE ES LO QUE ALIMENTA LA FASE 5
# ==============================================================================


def test_cada_decision_queda_registrada(cliente):
    antes = _filas_registradas()
    cliente.post("/predecir", json=cuerpo(solicitud_base(), solicitud_base()))
    assert _filas_registradas() == antes + 2


def test_el_registro_guarda_entrada_y_pronostico(cliente):
    cliente.post("/predecir", json=cuerpo(solicitud_base()))
    registro = pd.read_csv(md.RUTA_REGISTRO, sep=fe.SEPARADOR,
                           encoding=fe.ENCODING)
    for columna in ("momento", "origen", "version_modelo"):
        assert columna in registro.columns
    for columna in md.COLUMNAS_REQUERIDAS:
        assert columna in registro.columns
    for columna in ("decision", "probabilidad_mora", "puntaje"):
        assert columna in registro.columns


def test_el_registro_guarda_la_entrada_tal_como_llego(cliente):
    """Antes de sanear: el saneamiento borra la evidencia de como venia la
    fuente, y esa evidencia es una de las seniales de deriva.
    """
    s = solicitud_base()
    s["puntaje_datacredito"] = 0
    cliente.post("/predecir", json=cuerpo(s))
    registro = pd.read_csv(md.RUTA_REGISTRO, sep=fe.SEPARADOR,
                           encoding=fe.ENCODING)
    assert (registro["puntaje_datacredito"] == 0).any()


def _filas_registradas() -> int:
    if not md.RUTA_REGISTRO.exists():
        return 0
    return len(pd.read_csv(md.RUTA_REGISTRO, sep=fe.SEPARADOR,
                           encoding=fe.ENCODING))
