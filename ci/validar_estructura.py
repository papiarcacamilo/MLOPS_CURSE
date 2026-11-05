"""Compuerta de Jenkins: el repositorio conserva la estructura exigida.

El enunciado del Entregable 3 declara la estructura como no modificable, porque
el paso a produccion se valida contra ella. Este script hace esa comprobacion.

config.json, Base_de_datos.csv y requirements.txt se exigen en la raiz: es donde
los lee el proyecto, y asi queda declarado en el readme.

Solo usa la biblioteca estandar para correr antes de instalar dependencias: si la
estructura esta rota, la corrida se rechaza en segundos.

Uso: python ci/validar_estructura.py [--raiz RUTA]
Salida: 0 si cumple, 1 si no. Jenkins rechaza la corrida con cualquier valor
distinto de 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

CARPETA_PIPELINE = "mlops_pipeline/src"

ARCHIVOS_PIPELINE = (
    "Cargar_datos.ipynb",
    "comprension_eda.ipynb",
    "ft_engineering.py",
    "hueristic_model.py",
    "model_training.py",
    "model_deploy.py",
    "model_evaluation.py",
    "model_monitoring.py",
)

ARCHIVOS_RAIZ = (
    "config.json",
    "Base_de_datos.csv",
    "requirements.txt",
    ".gitignore",
    "readme.md",
    "set_up.bat",
)

REQUERIDOS = tuple(f"{CARPETA_PIPELINE}/{n}" for n in ARCHIVOS_PIPELINE) + ARCHIVOS_RAIZ


def revisar_estructura(raiz: Path) -> list[str]:
    """Devuelve los problemas encontrados; vacia si la estructura cumple."""
    problemas = []
    for relativa in REQUERIDOS:
        ruta = raiz / relativa
        if not ruta.is_file():
            problemas.append(f"falta {relativa}")
        elif ruta.stat().st_size == 0:
            problemas.append(f"esta vacio {relativa}")
    return problemas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raiz", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="raiz del repositorio a revisar")
    args = parser.parse_args(argv)

    problemas = revisar_estructura(args.raiz)
    for problema in problemas:
        print(f"[FALLA] {problema}")

    if problemas:
        print(f"RECHAZADO: {len(problemas)} de {len(REQUERIDOS)} archivos con problemas.")
        return 1
    print(f"APROBADO: estructura correcta, {len(REQUERIDOS)} archivos revisados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
