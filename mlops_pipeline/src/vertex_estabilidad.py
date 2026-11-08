"""
Entregable 5, Fase 2 - Estabilidad de la comparacion entre Python y AutoML
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/models/predicciones_test_comparacion.csv
Salida   : data/models/estabilidad_vertex.json
           data/models/figuras/vertex/estabilidad_bootstrap.png

Bootstrap pareado: se arman muchos tests nuevos sacando creditos del test real
con reposicion, y en cada uno se miden los dos modelos sobre LOS MISMOS
creditos. Asi la diferencia entre ellos no se mezcla con que a uno le tocaran
creditos mas faciles.

El remuestreo es estratificado: moras y no moras se sacan por separado para que
cada test simulado conserve la tasa de 4,74%. El AUC-PR depende de esa tasa, y
sin fijarla una parte de la variacion seria solo cambio de tasa.

Lectura: si el intervalo del 95% de la diferencia queda entero por encima de
cero, la ventaja de AutoML es firme; si incluye el cero, puede ser azar del test.

Uso:
    python mlops_pipeline/src/vertex_estabilidad.py
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import vertex_datos as vd
import vertex_evaluacion as ve

NOMBRE_JSON = "estabilidad_vertex.json"
NOMBRE_FIGURA = Path("figuras") / "vertex" / "estabilidad_bootstrap.png"
REMUESTREOS = 1000
SEMILLA = 42
METRICAS = {"auc_pr": average_precision_score, "roc_auc": roc_auc_score}


def indices_estratificados(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Un remuestreo con reposicion que conserva cuantas moras y no moras hay."""
    positivos = np.flatnonzero(y == 1)
    negativos = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(positivos, len(positivos), replace=True),
                           rng.choice(negativos, len(negativos), replace=True)])


def bootstrap_pareado(y, p_python, p_automl, remuestreos: int = REMUESTREOS,
                      semilla: int = SEMILLA) -> dict[str, np.ndarray]:
    """Cada metrica para los dos modelos, sobre los mismos creditos en cada muestra."""
    y = np.asarray(y).astype(int)
    p_python = np.asarray(p_python, dtype=float)
    p_automl = np.asarray(p_automl, dtype=float)
    rng = np.random.default_rng(semilla)
    valores = {f"{m}_{modelo}": np.empty(remuestreos)
               for m in METRICAS for modelo in ("python", "automl")}
    for i in range(remuestreos):
        idx = indices_estratificados(y, rng)
        for m, funcion in METRICAS.items():
            valores[f"{m}_python"][i] = funcion(y[idx], p_python[idx])
            valores[f"{m}_automl"][i] = funcion(y[idx], p_automl[idx])
    for m in METRICAS:
        valores[f"{m}_diferencia"] = valores[f"{m}_automl"] - valores[f"{m}_python"]
    return valores


def intervalo(valores: np.ndarray) -> dict:
    bajo, alto = np.percentile(valores, [2.5, 97.5])
    return {"media": round(float(np.mean(valores)), 4),
            "ic95": [round(float(bajo), 4), round(float(alto), 4)]}


def resumir(valores: dict[str, np.ndarray], ruido: float = ve.RUIDO_AUC_PR) -> dict:
    """Intervalos del 95% y la lectura de la diferencia en cada metrica."""
    resumen = {}
    for m in METRICAS:
        diferencia = valores[f"{m}_diferencia"]
        bajo = float(np.percentile(diferencia, 2.5))
        resumen[m] = {
            "python": intervalo(valores[f"{m}_python"]),
            "automl": intervalo(valores[f"{m}_automl"]),
            "diferencia": intervalo(diferencia),
            "pct_gana_automl": round(float((diferencia > 0).mean()) * 100, 1),
            "ventaja_firme": bool(bajo > 0),
        }
    resumen["auc_pr"]["pct_supera_el_ruido"] = round(
        float((valores["auc_pr_diferencia"] > ruido).mean()) * 100, 1)
    return resumen


def graficar(valores: dict[str, np.ndarray], resumen: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.5))
    for clave, color in (("python", "#1f6fb4"), ("automl", "#d1641b")):
        izq.hist(valores[f"auc_pr_{clave}"], bins=40, alpha=0.6, color=color,
                 label=f"{ve.MODELOS[clave]} (IC95 {resumen['auc_pr'][clave]['ic95']})")
    izq.set(title="AUC-PR en 1.000 tests remuestreados", xlabel="AUC-PR", ylabel="Muestras")
    diferencia = valores["auc_pr_diferencia"]
    bajo, alto = resumen["auc_pr"]["diferencia"]["ic95"]
    der.hist(diferencia, bins=40, color="#6b5ca5", alpha=0.75)
    der.axvline(0, color="black", linewidth=1.2, label="Sin diferencia")
    der.axvline(ve.RUIDO_AUC_PR, color="grey", linestyle="--",
                label=f"Ruido entre pliegues ({ve.RUIDO_AUC_PR})")
    der.axvspan(bajo, alto, color="#6b5ca5", alpha=0.12, label=f"IC95 [{bajo}, {alto}]")
    der.set(title="Diferencia de AUC-PR, AutoML menos Python", xlabel="Diferencia",
            ylabel="Muestras")
    for eje in (izq, der):
        eje.legend(fontsize=8)
        eje.grid(alpha=0.3)
    fig.suptitle("Bootstrap pareado y estratificado sobre el test de 2.153 creditos")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--salida", type=Path, default=ve.RUTA_MODELOS,
                        help="carpeta de salida; por defecto data/models")
    args = parser.parse_args(argv)

    predicciones = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV)
    y = predicciones[vd.OBJETIVO].to_numpy()
    valores = bootstrap_pareado(y, predicciones["p_python"], predicciones["p_automl"])
    resultado = {
        "metodo": {"remuestreos": REMUESTREOS, "semilla": SEMILLA,
                   "estratificado": True, "pareado": True},
        "bootstrap": resumir(valores),
    }
    args.salida.mkdir(parents=True, exist_ok=True)
    with open(args.salida / NOMBRE_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    graficar(valores, resultado["bootstrap"], args.salida / NOMBRE_FIGURA)

    for m, r in resultado["bootstrap"].items():
        print(f"{m}: Python {r['python']['ic95']} | AutoML {r['automl']['ic95']} | "
              f"diferencia {r['diferencia']['media']} {r['diferencia']['ic95']} | "
              f"gana AutoML en {r['pct_gana_automl']}% | firme: {r['ventaja_firme']}")
    print(f"La diferencia de AUC-PR supera el ruido en "
          f"{resultado['bootstrap']['auc_pr']['pct_supera_el_ruido']}% de las muestras.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
