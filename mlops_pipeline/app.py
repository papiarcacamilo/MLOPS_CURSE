"""
App de servicio del modelo
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : solicitudes de credito en originacion
Salida   : decision, probabilidad de mora, puntaje y motivo
Estado   : IMPLEMENTADA

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "... una app que permita disponibilizar dicho objeto y despliega el modelo
     en un endpoint que puede utilizarse para predicciones (por batch)."

Esta es la app. El motor de inferencia vive en model_deploy.py y la imagen que
contiene a ambos, en el Dockerfile. Aqui solo esta la capa HTTP: recibir,
delegar, responder.

POR QUE LA APP NO CALCULA NADA

Toda la logica de negocio esta en model_deploy.py. Si la app decidiera umbrales
o transformara datos, existirian dos caminos hacia la misma decision (el del
endpoint y el de un lote ejecutado desde consola) y con el tiempo divergirian.
Aqui la app traduce HTTP a DataFrame, llama a `predecir_lote` y traduce la
respuesta de vuelta.

EL MODELO SE CARGA UNA SOLA VEZ

Al arrancar, no en cada peticion. Deserializar el joblib por solicitud
multiplicaria la latencia sin ganar nada: el objeto es inmutable.

La cache vive en model_deploy.py, no aqui, para que un lote lanzado desde
consola se beneficie igual. El arranque solo la calienta.

ENDPOINTS

    GET  /salud             sonda de vida, para el contenedor
    GET  /modelo            metadatos: version, umbral, desempenio, esquema
    GET  /scorecard         tabla de puntos por tramo
    POST /predecir          lote en JSON
    POST /predecir/archivo  lote en CSV, respuesta en CSV
================================================================================
"""

from __future__ import annotations

import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    _DIR_MODULO = str(Path(__file__).resolve().parent)
except NameError:
    _DIR_MODULO = str(Path.cwd().resolve())
if _DIR_MODULO not in sys.path:
    sys.path.insert(0, _DIR_MODULO)

import ft_engineering as fe  # noqa: E402
import model_deploy as md  # noqa: E402

log = logging.getLogger("app")

# Tope del lote. Sin el, una peticion suficientemente grande agota la memoria
# del contenedor y tumba el servicio para todos los demas.
MAX_SOLICITUDES = 5000


class Solicitud(BaseModel):
    """Una solicitud de credito, tal como llega del sistema de originacion.

    Son las 13 columnas que el modelo necesita. No incluye los saldos (fuga:
    son posteriores al desembolso), ni `tipo_laboral` (descartada por medicion
    en la Fase 2 y eje de la auditoria de fairness), ni la variable objetivo,
    que en originacion todavia no existe.

    Los dos campos opcionales lo son por decision del EDA: su ausencia es
    informacion, y el WoE la trata como un tramo mas con su propio riesgo.
    """

    tipo_credito: int = Field(..., examples=[4])
    fecha_prestamo: str = Field(..., examples=["7/01/2025 14:40"],
                                description="Formato de origen d/m/Y")
    capital_prestado: float = Field(..., examples=[5_000_000])
    plazo_meses: int = Field(..., examples=[36])
    edad_cliente: int = Field(..., examples=[42])
    salario_cliente: float = Field(..., examples=[3_000_000])
    total_otros_prestamos: float = Field(..., examples=[12_000_000])
    cuota_pactada: float = Field(..., examples=[250_000])
    puntaje_datacredito: Optional[float] = Field(None, examples=[780])
    cant_creditosvigentes: int = Field(..., examples=[3])
    huella_consulta: int = Field(..., examples=[2])
    promedio_ingresos_datacredito: Optional[float] = Field(None, examples=[2_500_000])
    tendencia_ingresos: str = Field(..., examples=["Estable"])


class Lote(BaseModel):
    """Peticion por lote. Es el modo de uso que pide el enunciado."""

    solicitudes: list[Solicitud]
    origen: str = Field("api", description="Etiqueta que queda en el registro")


class Respuesta(BaseModel):
    """Una decision. Ningun campo queda vacio, ni siquiera en un rechazo."""

    decision: str
    probabilidad_mora: Optional[float]
    puntaje: Optional[float]
    banda: str
    motivo: str


# Estado del proceso. Se llena en el arranque para no releer artefactos en cada
# peticion.
ESTADO: dict = {}


@asynccontextmanager
async def ciclo_de_vida(_: FastAPI):
    """Carga los artefactos una vez y falla ruidosamente si falta alguno.

    Un contenedor que arranca sin modelo y devuelve errores 500 en produccion es
    peor que uno que no arranca: el primero parece sano.
    """
    ESTADO["artefacto"] = md.cargar_artefacto()
    ESTADO["scorecard"] = md.cargar_scorecard()
    ESTADO["version"] = md.version_modelo()
    # Deja el pipeline deserializado y caliente antes de la primera peticion.
    ESTADO["cadena"] = md.cadena_inferencia()
    log.info("Modelo %s cargado | umbral %.4f", ESTADO["version"],
             ESTADO["artefacto"]["umbral"]["valor"])
    yield
    ESTADO.clear()


app = FastAPI(
    title="MLOPS_CURSE | Riesgo crediticio",
    description=("Endpoint de prediccion por lote sobre el modelo seleccionado "
                 "en la Fase 3, evaluado una sola vez sobre test."),
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


@app.get("/salud")
def salud() -> dict:
    """Sonda de vida del contenedor."""
    return {
        "estado": "activo",
        "version_modelo": ESTADO.get("version"),
        "umbral": ESTADO["artefacto"]["umbral"]["valor"],
    }


@app.get("/modelo")
def metadatos() -> dict:
    """Que modelo responde, con que umbral y con que desempenio medido."""
    a = ESTADO["artefacto"]
    return {
        "modelo": a["modelo"],
        "desempenio_en_test": a["desempenio_en_test"],
        "umbral": a["umbral"],
        "politica": a["politica"],
        "esquema_entrada": a["esquema_entrada"],
        "bandas": a["bandas"],
    }


@app.get("/scorecard")
def scorecard() -> dict:
    """Tabla de puntos por tramo, con la que se justifica una negacion."""
    s = ESTADO["scorecard"]
    return {"escala": s["escala"], "puntos_base": s["puntos_base"],
            "tabla": s["tabla"], "ajustes_continuos": s["ajustes_continuos"],
            "nota": s["nota"]}


@app.post("/predecir", response_model=list[Respuesta])
def predecir(lote: Lote) -> list[dict]:
    """Predice un lote de solicitudes. Devuelve una decision por solicitud.

    El orden de la respuesta es el de la peticion, de modo que el llamador
    puede aparearlas por posicion sin necesidad de un identificador.
    """
    if not lote.solicitudes:
        raise HTTPException(400, "El lote no contiene solicitudes.")
    if len(lote.solicitudes) > MAX_SOLICITUDES:
        raise HTTPException(
            413, f"El lote excede el maximo de {MAX_SOLICITUDES} solicitudes.")

    datos = pd.DataFrame([s.model_dump() for s in lote.solicitudes])
    try:
        resultado = md.predecir_lote(datos, origen=lote.origen)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Una solicitud rechazada por contrato no tiene probabilidad ni puntaje, y
    # eso llega hasta aqui como NaN. JSON no admite NaN, asi que se pasa a null:
    # el `astype(object)` es obligatorio, porque en una columna float pandas
    # convierte None de vuelta a NaN sin avisar.
    return (resultado.astype(object)
            .where(pd.notna(resultado), None)
            .to_dict(orient="records"))


@app.post("/predecir/archivo")
async def predecir_archivo(archivo: UploadFile = File(...)) -> StreamingResponse:
    """Predice un lote entregado como CSV y responde con otro CSV.

    Es el modo de uso real de un proceso por batch: el area de riesgo deja un
    archivo con las solicitudes del dia y recoge el archivo de decisiones. El
    separador y la codificacion son los que declara config.json, los mismos del
    archivo de origen.
    """
    contenido = await archivo.read()
    try:
        datos = pd.read_csv(io.BytesIO(contenido), sep=fe.SEPARADOR,
                            encoding=fe.ENCODING)
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el CSV: {e}")

    if datos.empty:
        raise HTTPException(400, "El archivo no contiene solicitudes.")
    if len(datos) > MAX_SOLICITUDES:
        raise HTTPException(
            413, f"El archivo excede el maximo de {MAX_SOLICITUDES} solicitudes.")

    # Un archivo con otro esquema es culpa de quien llama. Sin este manejo, el
    # KeyError de la primera columna ausente salia como error 500, es decir,
    # como si el servicio estuviera roto.
    try:
        resultado = md.predecir_lote(datos, origen=f"archivo:{archivo.filename}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Se devuelven las decisiones junto a la entrada, para que el archivo de
    # salida sea legible sin tener que aparearlo con el de entrada.
    salida = pd.concat([datos.reset_index(drop=True), resultado], axis=1)
    buffer = io.StringIO()
    salida.to_csv(buffer, sep=fe.SEPARADOR, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=decisiones.csv"},
    )
