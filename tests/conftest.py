"""
Piezas compartidas por la suite
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

QUE PIDE LA ENTREGA 3

    Stage 2: "calidad, seguridad, cobertura, integridad y estilo".

Los tests cubren lo primero y lo tercero; SonarCloud, el resto.

POR QUE ESTOS TESTS Y NO OTROS

El proyecto lleva ocho fallos encontrados, y ninguno lanzaba una excepcion: la
banda fantasma en 650, las fechas sin dayfirst, el WoE que se iba a cero en
lotes pequenios, el saneamiento no idempotente, la linea base que se
reconstruia, el PSI sobre solicitudes repetidas, el esquema invalido que salia
como error 500 y el modelo que se releia en cada peticion.

Todos se encontraron mirando. La suite existe para que no haga falta mirar, y
sobre todo para que no vuelvan.

Ademas, las verificaciones que ya existian vivian dentro de `main()`, de modo que
solo corrian si alguien ejecutaba el script entero. Aqui salen de ahi.

LOS TESTS NO TOCAN EL REGISTRO DE PRODUCCION

`predecir_lote` escribe en data/monitoring/registro_endpoint.csv, que es el
insumo de la Fase 5. Una suite que lo ensucie falsearia la medicion de deriva,
asi que todo lo que puntua aqui lo hace con `guardar_registro=False`, y lo que no
puede elegir (el endpoint HTTP) escribe en un directorio temporal.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
PIPELINE = RAIZ / "mlops_pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import ft_engineering as fe  # noqa: E402
import model_deploy as md  # noqa: E402
import reglas_negocio as rn  # noqa: E402


# ==============================================================================
# ARTEFACTOS
# ==============================================================================
#
# Alcance de sesion: cargar el modelo y las particiones cuesta segundos, y no
# cambian durante la suite. Reabrirlos por test multiplicaria el tiempo sin
# aislar nada, porque ningun test los modifica.


@pytest.fixture(scope="session")
def raiz() -> Path:
    return RAIZ


@pytest.fixture(scope="session")
def artefacto() -> dict:
    """Contrato de servicio publicado por la Fase 4."""
    if not md.RUTA_ARTEFACTO.exists():
        pytest.skip("Falta artefacto_despliegue.json: corre model_deploy.py")
    return md.cargar_artefacto()


@pytest.fixture(scope="session")
def scorecard() -> dict:
    """Tabla de puntos publicada por la Fase 3."""
    if not md.RUTA_EVALUACION.exists():
        pytest.skip("Falta evaluacion.json: corre model_evaluation.py")
    return md.cargar_scorecard()


@pytest.fixture(scope="session")
def umbral(artefacto) -> float:
    return artefacto["umbral"]["valor"]


@pytest.fixture(scope="session")
def cadena():
    """Pipeline de servicio, ya ajustado."""
    return md.cadena_inferencia()


@pytest.fixture(scope="session")
def crudo() -> pd.DataFrame:
    """El archivo de origen, sin tocar."""
    ruta = RAIZ / fe.CONFIG["data"]["raw_path"]
    if not ruta.exists():
        pytest.skip(f"Falta {ruta.name}")
    return pd.read_csv(ruta, sep=fe.SEPARADOR, encoding=fe.ENCODING)


@pytest.fixture(scope="session")
def limpio() -> pd.DataFrame:
    """La salida de la Fase 1, que el saneamiento debe reproducir."""
    ruta = RAIZ / fe.CONFIG["data"]["clean_path"]
    if not ruta.exists():
        pytest.skip(f"Falta {ruta.name}")
    return pd.read_csv(ruta, sep=fe.SEPARADOR, encoding=fe.ENCODING)


@pytest.fixture(scope="session")
def train() -> pd.DataFrame:
    """Particion de entrenamiento de referencia."""
    ruta = fe.RUTA_SALIDA / "estratificado_train.csv"
    if not ruta.exists():
        pytest.skip("Faltan las particiones: corre ft_engineering.py")
    datos = pd.read_csv(ruta, sep=fe.SEPARADOR, encoding=fe.ENCODING)
    datos[fe.COLUMNA_FECHA] = pd.to_datetime(datos[fe.COLUMNA_FECHA])
    return datos


# ==============================================================================
# SOLICITUDES DE PRUEBA
# ==============================================================================


def solicitud_base() -> dict:
    """Una solicitud valida, con los valores del ejemplo del endpoint.

    Sirve de plantilla: cada test la copia y altera SOLO el campo que quiere
    probar, de modo que si falla se sabe que lo causo ese campo y no otro.
    """
    return {
        "tipo_credito": 4,
        "fecha_prestamo": "7/01/2025 14:40",
        "capital_prestado": 5_000_000,
        "plazo_meses": 36,
        "edad_cliente": 42,
        "salario_cliente": 3_000_000,
        "total_otros_prestamos": 12_000_000,
        "cuota_pactada": 250_000,
        "puntaje_datacredito": 780,
        "cant_creditosvigentes": 3,
        "huella_consulta": 2,
        "promedio_ingresos_datacredito": 2_500_000,
        "tendencia_ingresos": "Estable",
    }


@pytest.fixture
def solicitud() -> dict:
    return solicitud_base()


@pytest.fixture
def lote() -> pd.DataFrame:
    """Un lote de tres solicitudes validas y distintas entre si."""
    filas = []
    for score, plazo, edad in [(780, 36, 42), (610, 72, 24), (850, 12, 55)]:
        fila = solicitud_base()
        fila.update(puntaje_datacredito=score, plazo_meses=plazo,
                    edad_cliente=edad)
        filas.append(fila)
    return pd.DataFrame(filas)


@pytest.fixture(scope="session")
def contrato():
    return rn


@pytest.fixture(scope="session")
def despliegue():
    return md
