"""Compuerta de Jenkins: el modelo versionado cumple los minimos para aprobarse.

Lee lo que publico la evaluacion y lo compara con limites fijados aqui, a la
vista de quien revise el cambio. No reentrena ni vuelve a predecir: de medir se
encargan model_evaluation.py y los tests. Lo que si comprueba es que lo medido
corresponda al modelo guardado, para que no se apruebe un modelo con las metricas
de otro.

Hay dos niveles:
- Control: si no se cumple, la corrida se rechaza.
- Aviso: limitacion conocida. Se muestra en cada corrida, pero no rechaza, porque
  se acepto al aprobar el modelo. Rechazar por ella dejaria el pipeline siempre en
  rojo y nadie miraria el resultado.

Solo usa la biblioteca estandar, igual que validar_estructura.py.

Uso: python ci/validar_metricas.py [--raiz RUTA]
Salida: 0 si aprueba, 1 si no.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

RUTA_EVALUACION = "data/models/evaluacion.json"
RUTA_ARTEFACTO = "data/models/artefacto_despliegue.json"
RUTA_MODELO = "data/models/modelo_seleccionado.joblib"

# Variacion del AUC-PR entre pliegues en la seleccion del modelo. Por debajo de
# ella una diferencia se considera ruido; es la misma regla de empate del proyecto.
RUIDO_AUC_PR = 0.0159

# AUC-PR aprobado en la particion estratificada, menos el ruido. Bajar de aqui es
# una regresion real, no una variacion normal.
AUC_PR_MINIMO = round(0.1479 - RUIDO_AUC_PR, 4)

# Un test muy pequenio puede dar cualquier metrica: 2.000 creditos y 50 moras
# quedan por debajo de los dos test actuales (2.153 con 102, y 2.154 con 69).
N_TEST_MINIMO = 2000
EVENTOS_TEST_MINIMOS = 50

# 2 puntos porcentuales: cerca de 40% de la mora base (4,75%). Con un error mayor,
# las bandas de riesgo dejarian de significar lo que dicen.
ECE_MAXIMO = 0.02

# El umbral rechaza el mismo volumen que la regla heuristica, para poder compararlos.
PCT_RECHAZO_OBJETIVO = 20.9
PCT_RECHAZO_TOLERANCIA = 1.0

# Solo avisa: el KS del modelo aprobado ya esta por debajo.
KS_REFERENCIA = 0.30

_TOLERANCIA_IGUALDAD = 1e-6


class Control(NamedTuple):
    nombre: str
    aprobado: bool
    detalle: str


def _control(nombre: str, medir: Callable[[], tuple[bool, str]]) -> Control:
    """Un dato ausente o de otro tipo rechaza; nunca deja pasar ni rompe el script."""
    try:
        aprobado, detalle = medir()
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return Control(nombre, False, f"dato ausente o invalido: {error!r}")
    return Control(nombre, bool(aprobado), detalle)


def _brier_trivial(particion: dict) -> float:
    """Brier de predecir siempre la tasa base: el que debe mejorar el modelo."""
    tasa = particion["eventos_test"] / particion["n_test"]
    return tasa * (1 - tasa)


def _igual(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=_TOLERANCIA_IGUALDAD)


def revisar_controles(evaluacion: dict, artefacto: dict, version_real: str) -> list[Control]:
    """Reglas que rechazan. Se escriben como condicion de aprobar: un NaN no pasa."""

    def final(particion: str) -> dict:
        return evaluacion["evaluacion_final"][particion]

    def tamanio(particion: str) -> tuple[bool, str]:
        f = final(particion)
        return (f["n_test"] >= N_TEST_MINIMO and f["eventos_test"] >= EVENTOS_TEST_MINIMOS,
                f"{f['n_test']} creditos y {f['eventos_test']} moras; "
                f"minimo {N_TEST_MINIMO} y {EVENTOS_TEST_MINIMOS}")

    def supera_piso(particion: str) -> tuple[bool, str]:
        f = final(particion)
        modelo = f["metricas"]["auc_pr"]
        piso = f["piso_en_el_mismo_test"]["auc_pr"]
        return (modelo - piso >= RUIDO_AUC_PR,
                f"AUC-PR {modelo} contra piso {piso}; margen minimo {RUIDO_AUC_PR}")

    def auc_pr_minimo() -> tuple[bool, str]:
        valor = final("estratificado")["metricas"]["auc_pr"]
        return valor >= AUC_PR_MINIMO, f"{valor}; minimo {AUC_PR_MINIMO}"

    def brier_util() -> tuple[bool, str]:
        f = final("estratificado")
        valor, trivial = f["metricas"]["brier"], _brier_trivial(f)
        return valor < trivial, f"{valor}; prediccion trivial {trivial:.5f}"

    def ece_maximo() -> tuple[bool, str]:
        valor = final("estratificado")["calibracion"]["ece"]
        return valor <= ECE_MAXIMO, f"{valor}; maximo {ECE_MAXIMO}"

    def volumen_rechazo() -> tuple[bool, str]:
        valor = evaluacion["umbral"]["pct_rechazado"]
        return (abs(valor - PCT_RECHAZO_OBJETIVO) <= PCT_RECHAZO_TOLERANCIA,
                f"{valor}%; objetivo {PCT_RECHAZO_OBJETIVO} +/- {PCT_RECHAZO_TOLERANCIA}")

    def scorecard() -> tuple[bool, str]:
        coincide = evaluacion["verificacion_scorecard"]["coincide"]
        return coincide is True, f"coincide={coincide}"

    def version_modelo() -> tuple[bool, str]:
        publicada = artefacto["modelo"]["version"]
        return publicada == version_real, f"artefacto {publicada}; archivo {version_real}"

    def umbral_publicado() -> tuple[bool, str]:
        publicado = artefacto["umbral"]["valor"]
        evaluado = evaluacion["umbral"]["umbral"]
        return _igual(publicado, evaluado), f"artefacto {publicado}; evaluacion {evaluado}"

    def desempenio_publicado() -> tuple[bool, str]:
        publicado = artefacto["desempenio_en_test"]["auc_pr"]
        evaluado = final("estratificado")["metricas"]["auc_pr"]
        return _igual(publicado, evaluado), f"artefacto {publicado}; evaluacion {evaluado}"

    reglas: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("Tamanio del test estratificado", lambda: tamanio("estratificado")),
        ("Tamanio del test temporal", lambda: tamanio("temporal")),
        ("AUC-PR estratificado minimo", auc_pr_minimo),
        ("Supera al piso heuristico, estratificado", lambda: supera_piso("estratificado")),
        ("Supera al piso heuristico, temporal", lambda: supera_piso("temporal")),
        ("Brier mejor que el trivial, estratificado", brier_util),
        ("ECE estratificado maximo", ece_maximo),
        ("Volumen de rechazo del umbral", volumen_rechazo),
        ("Scorecard equivalente al modelo", scorecard),
        ("Version publicada igual al modelo guardado", version_modelo),
        ("Umbral publicado igual al evaluado", umbral_publicado),
        ("Desempenio publicado igual al evaluado", desempenio_publicado),
    ]
    return [_control(nombre, medir) for nombre, medir in reglas]


def revisar_avisos(evaluacion: dict) -> list[str]:
    """Limitaciones aceptadas al aprobar el modelo. No cambian el veredicto."""
    avisos = []
    try:
        finales = evaluacion["evaluacion_final"]
        for particion, f in finales.items():
            ks = f["metricas"]["ks"]
            if not ks >= KS_REFERENCIA:
                avisos.append(f"KS {particion} {ks}, por debajo de {KS_REFERENCIA}")

        temporal = finales["temporal"]
        brier, trivial = temporal["metricas"]["brier"], _brier_trivial(temporal)
        if not brier < trivial:
            avisos.append(f"Brier temporal {brier}, no mejora la prediccion trivial "
                          f"{trivial:.5f}")

        ece = temporal["calibracion"]["ece"]
        if not ece <= ECE_MAXIMO:
            avisos.append(f"ECE temporal {ece}, por encima de {ECE_MAXIMO}")
    except (KeyError, TypeError, AttributeError, ZeroDivisionError):
        # Los mismos datos los exigen los controles, que ya rechazan la corrida.
        pass
    return avisos


def _leer_json(ruta: Path) -> dict:
    with ruta.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raiz", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="raiz del repositorio a revisar")
    args = parser.parse_args(argv)

    try:
        evaluacion = _leer_json(args.raiz / RUTA_EVALUACION)
        artefacto = _leer_json(args.raiz / RUTA_ARTEFACTO)
        # Misma version que calcula model_deploy.version_modelo().
        version_real = hashlib.sha256(
            (args.raiz / RUTA_MODELO).read_bytes()).hexdigest()[:12]
    except (OSError, ValueError) as error:
        print(f"RECHAZADO: no se pudieron leer los artefactos: {error}")
        return 1

    controles = revisar_controles(evaluacion, artefacto, version_real)
    for control in controles:
        marca = "OK" if control.aprobado else "FALLA"
        print(f"[{marca}] {control.nombre}: {control.detalle}")
    for aviso in revisar_avisos(evaluacion):
        print(f"[AVISO] {aviso}")

    fallidos = sum(not c.aprobado for c in controles)
    if fallidos:
        print(f"RECHAZADO: {fallidos} de {len(controles)} controles no se cumplen.")
        return 1
    print(f"APROBADO: {len(controles)} controles cumplidos, modelo {version_real}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
