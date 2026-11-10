"""
Entregable 5, Fase 2 - Estabilidad de la comparacion entre Python y AutoML
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/models/predicciones_test_comparacion.csv y las particiones procesadas
Salida   : data/models/estabilidad_vertex.json
           data/models/figuras/vertex/estabilidad_bootstrap.png
           data/models/figuras/vertex/estabilidad_semillas.png
           data/models/figuras/vertex/estabilidad_tiempo.png

Bootstrap pareado: se arman muchos tests nuevos sacando creditos del test real
con reposicion, y en cada uno se miden los dos modelos sobre LOS MISMOS
creditos. Asi la diferencia entre ellos no se mezcla con que a uno le tocaran
creditos mas faciles.

El remuestreo es estratificado: moras y no moras se sacan por separado para que
cada test simulado conserve la tasa de 4,74%. El AUC-PR depende de esa tasa, y
sin fijarla una parte de la variacion seria solo cambio de tasa.

Lectura: si el intervalo del 95% de la diferencia queda entero por encima de
cero, la ventaja de AutoML es firme; si incluye el cero, puede ser azar del test.

Varias particiones: el bootstrap varia el test con el modelo fijo. Aqui se
separan los 10.763 creditos 20 veces, con otras semillas, y en cada una se
reentrena la misma receta. Mide cuanto depende el AUC-PR de Python de como se
separaron los datos. A AutoML no se le hace lo mismo: cada corrida cuesta
USD 21,25.

Consistencia en el tiempo: compara la particion estratificada con la temporal
usando medidas que no dependen de la tasa de mora, mide solo los creditos que
ya vencieron y entrena con el pasado para probar en ventanas de dos meses.
Un credito que aun no vence no puede estar en mora: en el test temporal el 53%
no habia vencido, asi que su tasa de mora esta subestimada.

Uso:
    python mlops_pipeline/src/vertex_estabilidad.py                       los dos analisis
    python mlops_pipeline/src/vertex_estabilidad.py --analisis tiempo     solo uno
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import vertex_datos as vd
import vertex_evaluacion as ve

NOMBRE_JSON = "estabilidad_vertex.json"
NOMBRE_FIGURA = Path("figuras") / "vertex" / "estabilidad_bootstrap.png"
NOMBRE_FIGURA_SEMILLAS = Path("figuras") / "vertex" / "estabilidad_semillas.png"
NOMBRE_FIGURA_TIEMPO = Path("figuras") / "vertex" / "estabilidad_tiempo.png"
COLUMNA_FECHA = "fecha_prestamo"
# Ventanas de dos meses en las que la mayoria de creditos ya vencio. Despues de
# agosto de 2025 casi todos siguen vigentes y la mora no se puede observar.
VENTANAS = (("2025-03-01", "2025-05-01"), ("2025-05-01", "2025-07-01"),
            ("2025-07-01", "2025-09-01"))
# La 42 es la de la particion original; aqui se prueban otras 20.
SEMILLAS = tuple(range(1, 21))
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


# ==============================================================================
# 5.2 | VARIAS PARTICIONES
# ==============================================================================


def datos_procesados(config: dict) -> pd.DataFrame:
    """Los 10.763 creditos procesados: el train y el test de la particion original."""
    return pd.concat([vd.leer_particion("estratificado_train", config),
                      vd.leer_particion("estratificado_test", config)], ignore_index=True)


def proporcion_test(carpeta: Path = vd.RUTA_PROCESADOS) -> float:
    with open(carpeta / "split_metadata.json", encoding="utf-8") as f:
        return float(json.load(f)["proporcion_test"])


def medir_particion(datos: pd.DataFrame, semilla: int, objetivo: str,
                    proporcion: float) -> dict:
    """Separa con otra semilla, reentrena la misma receta y la mide en su test."""
    train, test = train_test_split(datos, test_size=proporcion, stratify=datos[objetivo],
                                   random_state=semilla)
    receta = clone(joblib.load(ve.RUTA_MODELOS / "modelo_seleccionado.joblib"))
    receta.fit(train.drop(columns=[objetivo]), (1 - train[objetivo]).to_numpy())
    y = (1 - test[objetivo]).to_numpy()
    p = receta.predict_proba(test.drop(columns=[objetivo]))[:, 1]
    return {"semilla": semilla, "moras_test": int(y.sum()),
            "auc_pr": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4)}


def resumir_semillas(filas: list[dict], referencia: float, automl: float) -> dict:
    """Variacion del AUC-PR de Python entre particiones y donde caen los dos resultados."""
    auc = np.array([f["auc_pr"] for f in filas])
    return {
        "particiones": len(auc),
        "auc_pr": {"media": round(float(auc.mean()), 4),
                   "desviacion": round(float(auc.std(ddof=1)), 4),
                   "minimo": round(float(auc.min()), 4),
                   "maximo": round(float(auc.max()), 4)},
        "referencia_semilla_42": referencia,
        "pct_particiones_bajo_la_referencia": round(float((auc < referencia).mean()) * 100, 1),
        "automl_semilla_42": automl,
        "automl_supera_todas": bool(automl > auc.max()),
        "detalle": filas,
    }


def graficar_semillas(resumen: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auc = [f["auc_pr"] for f in resumen["detalle"]]
    fig, eje = plt.subplots(figsize=(9, 4.5))
    eje.scatter(range(1, len(auc) + 1), auc, color="#1f6fb4", zorder=3,
                label=f"Python en {len(auc)} particiones distintas")
    media = resumen["auc_pr"]["media"]
    eje.axhline(media, color="#1f6fb4", linestyle="--", label=f"Media de Python ({media})")
    eje.axhline(resumen["referencia_semilla_42"], color="black", linestyle=":",
                label=f"Python, particion original ({resumen['referencia_semilla_42']})")
    eje.axhline(resumen["automl_semilla_42"], color="#d1641b",
                label=f"AutoML, particion original ({resumen['automl_semilla_42']})")
    eje.set(title="AUC-PR de la misma receta al cambiar la particion",
            xlabel="Semilla de la particion", ylabel="AUC-PR en su test",
            xticks=range(1, len(auc) + 1))
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ==============================================================================
# 5.3 | CONSISTENCIA EN EL TIEMPO
# ==============================================================================


def resumen_particion(evaluacion: dict, particion: str) -> dict:
    """Metricas de una particion que no dependen de su tasa de mora, mas la tasa."""
    final = evaluacion["evaluacion_final"][particion]
    m = final["metricas"]
    resumen = {
        "creditos": final["n_test"], "moras": final["eventos_test"], "tasa": m["tasa_base"],
        "auc_pr": m["auc_pr"], "lift": m["lift_vs_azar"],
        "roc_auc": round((m["gini"] + 1) / 2, 4), "ks": m["ks"],
        "ventaja_sobre_el_piso": final["ventaja_sobre_el_piso"],
        "pct_sin_vencer": final["madurez_incompleta_pct"],
    }
    maduros = final["solo_maduros"]
    resumen["solo_vencidos"] = {"creditos": maduros["n"], "moras": maduros["eventos"],
                                "tasa": maduros["metricas"]["tasa_base"],
                                "auc_pr": maduros["metricas"]["auc_pr"],
                                "lift": maduros["metricas"]["lift_vs_azar"],
                                "ks": maduros["metricas"]["ks"]}
    return resumen


def medir_ventana(datos: pd.DataFrame, inicio: str, fin: str, objetivo: str) -> dict:
    """Entrena con todo lo anterior a `inicio` y mide en [inicio, fin)."""
    fecha = pd.to_datetime(datos[COLUMNA_FECHA])
    train = datos[fecha < inicio]
    test = datos[(fecha >= inicio) & (fecha < fin)]
    receta = clone(joblib.load(ve.RUTA_MODELOS / "modelo_seleccionado.joblib"))
    receta.fit(train.drop(columns=[objetivo]), (1 - train[objetivo]).to_numpy())
    y = (1 - test[objetivo]).to_numpy()
    p = receta.predict_proba(test.drop(columns=[objetivo]))[:, 1]
    auc_pr = float(average_precision_score(y, p))
    return {"ventana": f"{inicio} a {fin}", "creditos_train": int(len(train)),
            "creditos": int(len(test)), "moras": int(y.sum()),
            "tasa": round(float(y.mean()), 4), "auc_pr": round(auc_pr, 4),
            "lift": round(auc_pr / float(y.mean()), 2),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pct_sin_vencer": round(float(test["madurez_incompleta"].mean()) * 100, 1)}


def graficar_tiempo(resumen: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ventanas = resumen["ventanas"]
    etiquetas = [v["ventana"].replace(" a ", "\na ") for v in ventanas]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(ventanas))
    izq.plot(x, [v["lift"] for v in ventanas], marker="o", color="#1f6fb4", label="Lift en la ventana")
    izq.axhline(resumen["estratificado"]["lift"], color="grey", linestyle="--",
                label=f"Lift en el test estratificado ({resumen['estratificado']['lift']})")
    izq.axhline(1, color="black", linewidth=1, label="Azar")
    for i, v in zip(x, ventanas):
        izq.annotate(f"{v['moras']} moras", (i, v["lift"]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    izq.set(title="Lift sobre el azar, ventanas moviles", xticks=list(x), ylabel="AUC-PR / tasa",
            ylim=(0, max(v["lift"] for v in ventanas) * 1.3))
    izq.set_xticklabels(etiquetas, fontsize=8)
    # Eje de ROC-AUC fijo: con la escala automatica, 0,03 de diferencia parece enorme.
    der.plot(x, [v["roc_auc"] for v in ventanas], marker="o", color="#2a9d5c", label="ROC-AUC")
    der.axhline(resumen["estratificado"]["roc_auc"], color="grey", linestyle="--",
                label=f"ROC-AUC en el test estratificado ({resumen['estratificado']['roc_auc']})")
    der2 = der.twinx()
    der2.bar(x, [v["pct_sin_vencer"] for v in ventanas], alpha=0.2, color="#d1641b",
             label="% sin vencer, eje derecho")
    der.set(title="ROC-AUC y creditos sin vencer", xticks=list(x), ylabel="ROC-AUC",
            ylim=(0.5, 0.8))
    der2.set(ylabel="% de creditos sin vencer", ylim=(0, 100))
    der.set_xticklabels(etiquetas, fontsize=8)
    der.set_zorder(der2.get_zorder() + 1)
    der.patch.set_visible(False)
    izq.legend(loc="lower right", fontsize=8)
    lineas, nombres = der.get_legend_handles_labels()
    barras, nombres_barras = der2.get_legend_handles_labels()
    der.legend(lineas + barras, nombres + nombres_barras, loc="lower right", fontsize=8)
    for eje in (izq, der):
        eje.grid(alpha=0.3)
    fig.suptitle("La misma receta, entrenada con el pasado y probada en los meses siguientes")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--analisis", choices=("todos", "bootstrap", "semillas", "tiempo"),
                        default="todos", help="que analisis correr; por defecto todos")
    parser.add_argument("--salida", type=Path, default=ve.RUTA_MODELOS,
                        help="carpeta de salida; por defecto data/models")
    args = parser.parse_args(argv)

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_json = args.salida / NOMBRE_JSON
    resultado = (json.loads(ruta_json.read_text(encoding="utf-8"))
                 if ruta_json.exists() else {})
    predicciones = pd.read_csv(ve.RUTA_MODELOS / ve.NOMBRE_CSV)
    y = predicciones[vd.OBJETIVO].to_numpy()

    if args.analisis in ("todos", "bootstrap"):
        valores = bootstrap_pareado(y, predicciones["p_python"], predicciones["p_automl"])
        resultado["metodo"] = {"remuestreos": REMUESTREOS, "semilla": SEMILLA,
                               "estratificado": True, "pareado": True}
        resultado["bootstrap"] = resumir(valores)
        graficar(valores, resultado["bootstrap"], args.salida / NOMBRE_FIGURA)
        for m, r in resultado["bootstrap"].items():
            print(f"{m}: Python {r['python']['ic95']} | AutoML {r['automl']['ic95']} | "
                  f"diferencia {r['diferencia']['media']} {r['diferencia']['ic95']} | "
                  f"gana AutoML en {r['pct_gana_automl']}% | firme: {r['ventaja_firme']}")
        print(f"La diferencia de AUC-PR supera el ruido en "
              f"{resultado['bootstrap']['auc_pr']['pct_supera_el_ruido']}% de las muestras.")

    if args.analisis in ("todos", "semillas"):
        config = vd.cargar_config()
        datos = datos_procesados(config)
        filas = [medir_particion(datos, s, config["target_variable"], proporcion_test())
                 for s in SEMILLAS]
        referencia = round(float(average_precision_score(y, predicciones["p_python"])), 4)
        automl = round(float(average_precision_score(y, predicciones["p_automl"])), 4)
        resultado["semillas"] = resumir_semillas(filas, referencia, automl)
        graficar_semillas(resultado["semillas"], args.salida / NOMBRE_FIGURA_SEMILLAS)
        r = resultado["semillas"]
        print(f"AUC-PR de Python en {r['particiones']} particiones: media {r['auc_pr']['media']}, "
              f"desviacion {r['auc_pr']['desviacion']}, rango [{r['auc_pr']['minimo']}, "
              f"{r['auc_pr']['maximo']}]")
        print(f"La particion original ({referencia}) supera al "
              f"{r['pct_particiones_bajo_la_referencia']}% de las otras; AutoML ({automl}) "
              f"supera a todas: {r['automl_supera_todas']}")

    if args.analisis in ("todos", "tiempo"):
        config = vd.cargar_config()
        with open(ve.RUTA_MODELOS / "evaluacion.json", encoding="utf-8") as f:
            evaluacion = json.load(f)
        datos = datos_procesados(config)
        resultado["tiempo"] = {
            "estratificado": resumen_particion(evaluacion, "estratificado"),
            "temporal": resumen_particion(evaluacion, "temporal"),
            "ventanas": [medir_ventana(datos, inicio, fin, config["target_variable"])
                         for inicio, fin in VENTANAS],
        }
        graficar_tiempo(resultado["tiempo"], args.salida / NOMBRE_FIGURA_TIEMPO)
        t = resultado["tiempo"]
        for nombre in ("estratificado", "temporal"):
            r = t[nombre]
            print(f"{nombre}: AUC-PR {r['auc_pr']} con tasa {r['tasa']} | lift {r['lift']} | "
                  f"ROC-AUC {r['roc_auc']} | {r['pct_sin_vencer']}% sin vencer | solo vencidos: "
                  f"lift {r['solo_vencidos']['lift']} con {r['solo_vencidos']['moras']} moras")
        for v in t["ventanas"]:
            print(f"ventana {v['ventana']}: {v['moras']} moras | lift {v['lift']} | "
                  f"ROC-AUC {v['roc_auc']} | {v['pct_sin_vencer']}% sin vencer")

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
