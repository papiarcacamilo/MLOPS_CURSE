"""
Monitoreo y medida de data drift
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/monitoring/registro_endpoint.csv, que produce model_deploy.py
Salida   : metricas de deriva con periodicidad definida, y su linea base
Estado   : IMPLEMENTADA

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Crea el trabajo de monitoreo que trae en una tabla los datos pasados al
     endpoint junto con los pronosticos entregados por este y los utiliza, con
     una periodicidad definida, para muestrear y obtener metricas que permitan
     detectar cambios en la poblacion que puedan afectar el desempenio del
     modelo. Medida del Data drift."

La tabla ya existe: `registro_endpoint.csv` guarda las 13 columnas de entrada
tal como llegaron, mas la decision, la probabilidad, el puntaje y la version del
modelo que respondio. Aqui se lee, se muestrea por periodo y se mide.

POR QUE ESTE PROYECTO TIENE UN CASO DE DERIVA REAL

No es un ejercicio teorico. La Fase 1 documento que la tasa de mora de esta
cartera oscila entre 1.72% y 9.09% segun el mes de desembolso. Un modelo
entrenado sobre la mezcla completa vera poblaciones que se desvian de esa media
con regularidad, y la particion temporal ya lo confirmo: entrenando con el
pasado y validando con el futuro, la estructura cambia.

LOS CORTES DEL PSI SE IMPORTAN DE LA RECETA

Y no se recalculan. Es la misma regla de la Fase 4: si el PSI usa tramos
distintos a los del binning, mide una cosa y el modelo ve otra. Con los cortes de
la receta, un PSI alto en `puntaje_datacredito` significa literalmente que los
tramos que el modelo usa se estan llenando distinto, que es la unica lectura
accionable.

CUATRO SENIALES, Y NO LLEGAN A LA VEZ

    covariables    distribucion de cada entrada contra su linea base. PSI por
                   tramo. Se mide de inmediato
    prediccion     distribucion de las probabilidades emitidas. Se detecta sin
                   esperar la etiqueta, y por eso es la senial mas temprana
    corte          si el punto de operacion sigue rechazando el ~21% que
                   rechazaba, o la poblacion se ha movido bajo el umbral fijo
    concepto       relacion entre entradas y resultado real. Requiere que la
                   etiqueta madure, asi que llega tarde por definicion

LA TRAMPA DE ESTE DATASET, Y NO ES LA QUE PARECIA

El 19.9% de los creditos tiene madurez incompleta; en el test temporal, el
53.4%. El razonamiento habitual dice que un credito recien desembolsado aparece
como "al dia" solo porque no ha transcurrido el plazo, y que por tanto medir
sobre creditos jovenes subestima la mora. Este proyecto lo repitio en varias
fases. Medido sobre el test temporal, no es lo que pasa:

    solo vencidos   n=1003  plazo medio  6.0 meses  mora 2.09%
    solo jovenes    n=1151  plazo medio 15.8 meses  mora 4.17%

Los jovenes tienen el DOBLE de mora, no la mitad. La razon es que
`madurez_incompleta` esta confundida con el plazo: un credito sigue vivo
justamente porque se pacto a mas meses, y `plazo_meses` es el coeficiente mas
alto del modelo (0.8573). El filtro de madurez no aisla el censurado, selecciona
creditos cortos, que son estructuralmente menos riesgosos.

Conclusion practica: NINGUNO de los dos subgrupos da una lectura limpia. El
censurado existe y empuja hacia abajo; la composicion por plazo existe y empuja
hacia arriba, y aqui gana la segunda. Por eso `deriva_concepto` reporta la
cohorte completa y el subgrupo vencido POR SEPARADO, con lift ademas de AUC-PR,
que es lo unico comparable entre tasas base distintas.

EL EXPERIMENTO TIENE CONTROL

Medir y anunciar "hay deriva" no prueba nada si la medicion nunca se contrasto.
`main()` evalua DOS cohortes:

    control    muestra de estratificado_train, la misma poblacion del ajuste.
               Debe dar PSI cercano a cero. Si no lo da, la medida esta rota y
               el otro numero no merece credito
    reciente   temporal_test, creditos desembolsados desde 2025-07-12. Es una
               poblacion genuinamente posterior, con mora 3.20% frente al 5.13%
               de su train

LINEA BASE COMO ARTEFACTO

Se calcula UNA vez sobre train y se congela en
data/models/linea_base_monitoreo.json. Construirla exige los datos de
entrenamiento; MEDIR contra ella, no. Publicada ya, comparar una cohorte solo
necesita ese JSON, y por eso `cargar_linea_base()` la lee en lugar de rehacerla.

Reconstruirla es explicito, con `--reconstruir-base`, y solo hace falta cuando
cambia el modelo: una linea base pertenece a la version que la genero.

El monitoreo corre fuera de la imagen. La de la Fase 4 lleva unicamente lo que
el endpoint necesita para responder: ni matplotlib, ni las particiones, ni este
modulo.

EL MONITOREO NO FABRICA EL TRAFICO QUE MIDE

Las cohortes de contraste se registran la PRIMERA vez y despues solo se puntuan.
Registrarlas en cada corrida hacia crecer el registro sin limite con las mismas
solicitudes, y un registro inflado por el propio medidor no describe a nadie.
================================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    _DIR_MODULO = str(Path(__file__).resolve().parent)
except NameError:
    _DIR_MODULO = str(Path.cwd().resolve())
if _DIR_MODULO not in sys.path:
    sys.path.insert(0, _DIR_MODULO)

import ft_engineering as fe  # noqa: E402
import model_deploy as md  # noqa: E402
import reglas_negocio as rn  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("monitoreo")

RUTA_RAIZ = fe.RUTA_RAIZ
RUTA_MODELOS = fe.RUTA_MODELOS
RUTA_DATOS = fe.RUTA_SALIDA
RUTA_MONITOREO = md.RUTA_MONITOREO

RUTA_REGISTRO = md.RUTA_REGISTRO
RUTA_LINEA_BASE = RUTA_MODELOS / "linea_base_monitoreo.json"
RUTA_INFORME = RUTA_MODELOS / "monitoreo.json"
RUTA_FIGURAS = RUTA_MODELOS / "figuras" / "monitoreo"

TARGET = fe.TARGET
SEMILLA = fe.SEMILLA


# ==============================================================================
# PERIODICIDAD Y UMBRALES
# ==============================================================================

# Periodicidad del muestreo. Semanal, y no diaria, por el volumen: con lotes de
# originacion un dia puede traer pocas decenas de solicitudes, y un PSI sobre 30
# registros es ruido con nombre tecnico. Semanal acumula lo suficiente sin
# retrasar la deteccion mas de lo aceptable en un producto de credito, donde una
# decision tarda dias en ejecutarse de todos modos.
PERIODICIDAD = "W"
NOMBRE_PERIODICIDAD = "semanal"

# Segundo eje: la cosecha de desembolso. Mensual, porque es la unidad sobre la
# que la Fase 1 midio que la mora varia (de 1.72% a 9.09% segun el mes) y porque
# es el analisis de vintages con el que trabaja cualquier area de riesgo.
PERIODICIDAD_COSECHA = "MS"
NOMBRE_PERIODICIDAD_COSECHA = "mensual por cosecha de desembolso"

# Tamanio minimo por debajo del cual la medida NO se publica. Es preferible un
# hueco declarado a un numero que nadie puede interpretar.
MIN_MUESTRA = 100

# Tope del muestreo por periodo. Por encima de este volumen se toma una muestra
# aleatoria con la semilla del proyecto: el PSI se estabiliza mucho antes de los
# 5.000 registros y procesar el lote entero solo cuesta tiempo.
MAX_MUESTRA = 5000

# Umbrales del sector para el PSI (Population Stability Index). Son la
# convencion en scorecards de credito, no una eleccion de este proyecto.
UMBRAL_PSI_MODERADO = 0.10
UMBRAL_PSI_IMPORTANTE = 0.25

# Correccion para tramos vacios. Sin ella el PSI se va a infinito en cuanto un
# tramo se queda sin observaciones, que con muestras pequenias pasa a menudo.
EPSILON = 1e-6

# Variables monetarias: no tienen cortes de negocio, de modo que la linea base
# los fija por cuantiles sobre train y los congela.
N_TRAMOS_MONETARIAS = 5

ESTADO_ESTABLE = "estable"
ESTADO_MODERADO = "cambio moderado"
ESTADO_IMPORTANTE = "cambio importante"


# ==============================================================================
# 1 | LA TABLA DEL ENDPOINT
# ==============================================================================


def cargar_registro(ruta: Path = None) -> pd.DataFrame:
    """Lee la tabla de entradas y pronosticos que produce el endpoint."""
    ruta = ruta or RUTA_REGISTRO
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta.name}. El monitoreo se alimenta del endpoint: "
            f"ejecuta primero\n    python mlops_pipeline/model_deploy.py")

    registro = pd.read_csv(ruta, sep=fe.SEPARADOR, encoding=fe.ENCODING)
    registro["momento"] = pd.to_datetime(registro["momento"], format="ISO8601")
    return registro


def preparar(datos: pd.DataFrame, crudo: bool | None = None) -> pd.DataFrame:
    """Reconstruye las columnas que el modelo ve, sanee o no haga falta.

    El registro guarda la entrada TAL COMO LLEGO, antes de sanear, porque el
    saneamiento borra la evidencia de como venia la fuente. Para medir deriva
    sobre lo que el modelo realmente consume hay que rehacer ese camino, y se
    rehace llamando al mismo codigo del endpoint en lugar de reimplementarlo.

    Pero las particiones de data/processed/ YA vienen saneadas, y volver a
    sanearlas las destruye: `sanear` recalcula las banderas comparando contra los
    umbrales, y sobre valores ya corregidos ninguna se activa. Medido sobre
    train, la segunda pasada apagaba 103 banderas de edad, 187 de salario y 47 de
    tendencia, y rellenaba con la mediana los 105 nulos intencionales del score.
    Esa corrupcion silenciosa fue lo que hizo fallar al control: 0.1153 de PSI
    sobre la propia poblacion de ajuste, donde deberia dar cero.

    `crudo=None` decide por la presencia de las banderas, que solo existen si el
    saneamiento ya corrio.
    """
    if crudo is None:
        crudo = not set(fe.VARS_BINARIAS).issubset(datos.columns)

    if crudo:
        datos = md.sanear(datos)
    else:
        datos = datos.copy()
        if not pd.api.types.is_datetime64_any_dtype(datos[fe.COLUMNA_FECHA]):
            datos[fe.COLUMNA_FECHA] = pd.to_datetime(datos[fe.COLUMNA_FECHA])

    return fe.construir_derivadas(datos, md.FECHA_CORTE_OBS)


def muestrear(datos: pd.DataFrame, columna: str = "momento",
              periodicidad: str = PERIODICIDAD) -> dict:
    """Parte el registro en periodos y devuelve una muestra de cada uno.

    `columna` elige el eje del periodo, y los dos que admite responden preguntas
    distintas:

        momento          cuando llego la peticion. Es la vista OPERATIVA: dice si
                         lo que esta entrando hoy se parece a lo que el modelo
                         aprendio
        fecha_prestamo   cuando se desembolso el credito. Es la vista de COSECHA,
                         el analisis de vintages estandar en riesgo. La Fase 1
                         midio que la mora de esta cartera va de 1.72% a 9.09%
                         segun el mes de desembolso, de modo que la cosecha es el
                         periodo sobre el que el riesgo realmente varia

    Los periodos por debajo de `MIN_MUESTRA` se devuelven igual, marcados como no
    publicables: que un periodo tenga poco volumen es en si mismo una senial que
    el operador debe ver, no algo que convenga esconder.
    """
    datos = datos.copy()
    if not pd.api.types.is_datetime64_any_dtype(datos[columna]):
        # `_a_fecha` del contrato, no `pd.to_datetime` a secas: el registro
        # guarda `fecha_prestamo` en el formato de origen d/m/Y, y dejar que
        # pandas lo adivine desplaza el rango del dataset entero. La Fase 1 ya
        # pago ese error una vez, con 21 "fechas futuras" que no existian.
        datos[columna] = rn._a_fecha(datos[columna], dayfirst=True)
    datos = datos[datos[columna].notna()]

    columnas = [c for c in md.COLUMNAS_REQUERIDAS if c in datos.columns]
    periodos = {}

    for etiqueta, grupo in datos.groupby(
            pd.Grouper(key=columna, freq=periodicidad)):
        if grupo.empty:
            continue

        # Una misma solicitud puntuada dos veces es UNA observacion de la
        # poblacion, no dos. Sin este paso, reenviar un lote duplicaria su peso
        # en el PSI y disipararia una alarma falsa: medido sobre el registro de
        # las pruebas del endpoint, 250 decisiones eran 50 solicitudes reales, y
        # el PSI de prediccion daba 0.2858, es decir "reentrenar", sobre una
        # poblacion que no habia cambiado en absoluto.
        distintas = grupo.drop_duplicates(subset=columnas) if columnas else grupo

        muestra = (distintas.sample(MAX_MUESTRA, random_state=SEMILLA)
                   if len(distintas) > MAX_MUESTRA else distintas)
        periodos[str(etiqueta.date())] = {
            "n_decisiones": int(len(grupo)),
            "n_solicitudes": int(len(distintas)),
            "n_muestra": int(len(muestra)),
            "suficiente": bool(len(distintas) >= MIN_MUESTRA),
            "datos": muestra,
        }
    return periodos


# ==============================================================================
# 2 | PSI
# ==============================================================================


def distribucion(serie: pd.Series, tramos: list | None,
                 categorias: list | None = None) -> np.ndarray:
    """Proporcion de la serie en cada tramo. Los nulos son un tramo mas.

    Tratarlos como categoria propia, y no descartarlos, es coherente con el
    binning: alli la ausencia de dato tiene su propio WoE porque la Fase 1
    decidio que ausencia es informacion. Si el monitoreo los ignorase, un lote
    que de pronto llega sin score no dispararia ninguna alarma.
    """
    if categorias is not None:
        conteo = serie.astype(str).where(serie.notna(), fe.ETIQUETA_NULO)
        etiquetas = list(categorias) + [fe.ETIQUETA_NULO]
        frecuencias = conteo.value_counts().reindex(etiquetas, fill_value=0)
    else:
        cortes = [float(c) for c in tramos]
        conteo = pd.cut(serie, cortes).astype(str)
        conteo = conteo.where(serie.notna(), fe.ETIQUETA_NULO)
        etiquetas = sorted(set(conteo.unique()) | {fe.ETIQUETA_NULO})
        frecuencias = conteo.value_counts().reindex(etiquetas, fill_value=0)

    total = frecuencias.sum()
    if total == 0:
        return np.zeros(len(frecuencias)), list(frecuencias.index)
    return (frecuencias / total).to_numpy(), list(frecuencias.index)


def psi(esperado: dict, observado: dict) -> float:
    """Population Stability Index entre dos distribuciones por tramo.

        PSI = suma( (obs - esp) * ln(obs / esp) )

    Es simetrico y siempre positivo. Cero significa distribuciones identicas.
    Los tramos se aparean por NOMBRE, no por posicion: si la cohorte no tiene
    algun tramo, ese entra con proporcion cero mas epsilon en vez de
    desalinear todo el vector.
    """
    etiquetas = sorted(set(esperado) | set(observado))
    esp = np.array([esperado.get(t, 0.0) for t in etiquetas]) + EPSILON
    obs = np.array([observado.get(t, 0.0) for t in etiquetas]) + EPSILON
    esp, obs = esp / esp.sum(), obs / obs.sum()
    return float(np.sum((obs - esp) * np.log(obs / esp)))


def clasificar(valor: float) -> str:
    """Traduce el PSI al semaforo del sector."""
    if valor < UMBRAL_PSI_MODERADO:
        return ESTADO_ESTABLE
    if valor < UMBRAL_PSI_IMPORTANTE:
        return ESTADO_MODERADO
    return ESTADO_IMPORTANTE


# ==============================================================================
# 3 | LINEA BASE
# ==============================================================================


def _tramos_de_la_receta() -> dict:
    """Cortes y categorias del binning, tal como los dejo la Fase 2."""
    with open(RUTA_DATOS / "receta_estratificado.json", encoding="utf-8") as f:
        receta = json.load(f)["binning_woe"]

    tramos = {}
    for variable, detalle in receta.items():
        if detalle.get("cortes"):
            tramos[variable] = {"tipo": "numerica", "cortes": detalle["cortes"]}
        else:
            categorias = [c for c in detalle["woe"] if c != fe.ETIQUETA_NULO]
            tramos[variable] = {"tipo": "categorica", "categorias": categorias}
    return tramos


def construir_linea_base() -> dict:
    """Congela la distribucion de la poblacion de entrenamiento.

    Se calcula una vez y se guarda. Recalcularla en cada corrida tendria el
    mismo problema que recalcular las medianas del saneamiento: la referencia se
    movria con los datos, y una deriva lenta nunca se detectaria porque la vara
    de medir se desplazaria junto con lo medido.
    """
    train = pd.read_csv(RUTA_DATOS / "estratificado_train.csv",
                        sep=fe.SEPARADOR, encoding=fe.ENCODING)
    train["fecha_prestamo"] = pd.to_datetime(train["fecha_prestamo"])

    tramos = _tramos_de_la_receta()
    variables = {}

    # Las 9 del binning, con los cortes del modelo.
    for variable, detalle in tramos.items():
        if detalle["tipo"] == "numerica":
            props, etiquetas = distribucion(train[variable], detalle["cortes"])
        else:
            props, etiquetas = distribucion(train[variable], None,
                                            detalle["categorias"])
        variables[variable] = {
            **detalle,
            "distribucion": dict(zip(etiquetas, np.round(props, 6).tolist())),
        }

    # Las 4 monetarias, por cuantiles congelados.
    for variable in fe.VARS_MONETARIAS:
        bordes = np.quantile(train[variable].dropna(),
                             np.linspace(0, 1, N_TRAMOS_MONETARIAS + 1))
        cortes = [-np.inf] + [float(b) for b in np.unique(bordes)[1:-1]] + [np.inf]
        props, etiquetas = distribucion(train[variable], cortes)
        variables[variable] = {
            "tipo": "numerica",
            "cortes": cortes,
            "distribucion": dict(zip(etiquetas, np.round(props, 6).tolist())),
        }

    # Las 4 banderas, por tasa de activacion.
    banderas = {}
    for bandera in fe.VARS_BINARIAS:
        activa = train[bandera].astype(str).str.lower().isin(["true", "1"])
        banderas[bandera] = round(float(activa.mean()), 6)

    # La distribucion de la probabilidad que el modelo emite sobre su propia
    # poblacion de ajuste, y el volumen de rechazo que le corresponde.
    cadena = md.cadena_inferencia()
    p_train = cadena.predict_proba(train.drop(columns=[TARGET]))[:, 1]
    bordes_p = np.quantile(p_train, np.linspace(0, 1, 11))
    cortes_p = [-np.inf] + [float(b) for b in np.unique(bordes_p)[1:-1]] + [np.inf]
    props_p, etiquetas_p = distribucion(pd.Series(p_train), cortes_p)

    umbral = md.cargar_artefacto()["umbral"]["valor"]

    return {
        "origen": "estratificado_train.csv",
        "n": int(len(train)),
        "version_modelo": md.version_modelo(),
        "periodicidad": NOMBRE_PERIODICIDAD,
        "umbrales_psi": {"moderado": UMBRAL_PSI_MODERADO,
                         "importante": UMBRAL_PSI_IMPORTANTE},
        "variables": variables,
        "banderas": banderas,
        "prediccion": {
            "cortes": cortes_p,
            "distribucion": dict(zip(etiquetas_p, np.round(props_p, 6).tolist())),
            "probabilidad_media": round(float(p_train.mean()), 6),
            "pct_rechazo": round(float((p_train >= umbral).mean()) * 100, 2),
            "umbral": umbral,
        },
        "nota_cortes": ("Los cortes de las 9 variables del binning se importan de "
                        "receta_estratificado.json. Medir el PSI sobre otros "
                        "tramos daria un numero que no corresponde a lo que el "
                        "modelo ve."),
    }


def guardar_linea_base(base: dict) -> Path:
    """Publica la linea base en disco."""
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    with open(RUTA_LINEA_BASE, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2, ensure_ascii=False)
    return RUTA_LINEA_BASE


def cargar_linea_base(reconstruir: bool = False) -> tuple[dict, bool]:
    """Devuelve la linea base, y si hubo que construirla.

    CONGELADA quiere decir que se construye UNA vez y despues solo se lee. La
    primera version de este modulo la reconstruia en cada corrida, que es
    exactamente el error contra el que advierte su propio docstring: si la
    referencia se recalcula sobre unos datos de entrenamiento que alguien
    regenero, se desplaza en silencio junto con lo medido y una deriva lenta no
    se detecta jamas.

    Tambien es lo que hace cierta la otra promesa: MEDIR no necesita los datos
    de entrenamiento, solo este JSON. Reconstruir si.

    `reconstruir=True` es la unica forma de rehacerla, y existe para cuando el
    modelo cambia: una linea base pertenece a la version del modelo que la
    genero, no al proyecto.
    """
    if not reconstruir and RUTA_LINEA_BASE.exists():
        with open(RUTA_LINEA_BASE, encoding="utf-8") as f:
            return json.load(f), False

    base = construir_linea_base()
    guardar_linea_base(base)
    return base, True


# ==============================================================================
# 4 | LAS CUATRO SENIALES
# ==============================================================================


def deriva_covariables(cohorte: pd.DataFrame, base: dict) -> pd.DataFrame:
    """PSI por variable de entrada, ordenado de mayor a menor."""
    filas = []
    for variable, detalle in base["variables"].items():
        if variable not in cohorte.columns:
            continue
        if detalle["tipo"] == "numerica":
            props, etiquetas = distribucion(cohorte[variable], detalle["cortes"])
        else:
            props, etiquetas = distribucion(cohorte[variable], None,
                                            detalle["categorias"])
        valor = psi(detalle["distribucion"], dict(zip(etiquetas, props)))
        filas.append({"variable": variable, "psi": round(valor, 4),
                      "estado": clasificar(valor)})

    tabla = pd.DataFrame(filas).sort_values("psi", ascending=False)
    return tabla.reset_index(drop=True)


def deriva_prediccion(probabilidades: np.ndarray, base: dict) -> dict:
    """PSI sobre la distribucion de probabilidades emitidas.

    Es la senial mas temprana que existe: no espera a la etiqueta. Si la
    poblacion cambia de forma que afecte al modelo, sus probabilidades se
    desplazan antes de que ningun credito venza.
    """
    referencia = base["prediccion"]
    props, etiquetas = distribucion(pd.Series(probabilidades),
                                    referencia["cortes"])
    valor = psi(referencia["distribucion"], dict(zip(etiquetas, props)))
    media = float(np.mean(probabilidades))

    return {
        "psi": round(valor, 4),
        "estado": clasificar(valor),
        "probabilidad_media": round(media, 6),
        "probabilidad_media_base": referencia["probabilidad_media"],
        "desplazamiento": round(media - referencia["probabilidad_media"], 6),
    }


def estabilidad_del_corte(probabilidades: np.ndarray, base: dict) -> dict:
    """Comprueba que el umbral fijo siga rechazando el volumen previsto.

    El umbral 0.0661 se fijo para rechazar el 20.9% de la poblacion de ajuste.
    Es un numero FIJO sobre una poblacion que se mueve, de modo que el volumen
    rechazado deriva aunque el modelo no cambie. La evaluacion ya lo habia
    detectado: 20.9% fuera de fold, 21.6% en el test estratificado y 33.3% en el
    temporal.

    Por eso se vigila el volumen, no solo el PSI: un salto del rechazo es una
    consecuencia de negocio inmediata, con o sin deriva estadistica declarada.
    """
    referencia = base["prediccion"]
    umbral = referencia["umbral"]
    observado = float((probabilidades >= umbral).mean()) * 100
    esperado = referencia["pct_rechazo"]

    return {
        "umbral": umbral,
        "pct_rechazo_base": esperado,
        "pct_rechazo_cohorte": round(observado, 2),
        "desviacion_pp": round(observado - esperado, 2),
        # Un tercio de desviacion relativa cambia el volumen de operacion de un
        # area de riesgo, y eso se revisa aunque el PSI no llegue a 0.10.
        "requiere_revision": bool(abs(observado - esperado) > esperado / 3),
    }


def deriva_banderas(cohorte: pd.DataFrame, base: dict) -> pd.DataFrame:
    """Tasa de activacion de las 4 banderas contra su linea base.

    Se vigilan aparte porque una bandera no es una distribucion: es una tasa, y
    un PSI sobre dos tramos oculta lo que importa.

    `tendencia_ingresos_reconstruida` merece atencion especial. En train tiene 47
    casos y CERO moras, de modo que su coeficiente (-3.9728) no esta estimado
    sino determinado por la separacion perfecta, y en el scorecard regala 115
    puntos sobre un rango de 302. Si su tasa de activacion sube, el modelo
    empieza a regalar puntos a mas gente por una relacion que nunca se pudo
    medir. Es la limitacion declarada en la evaluacion, y aqui es donde se
    vigila.
    """
    filas = []
    for bandera, tasa_base in base["banderas"].items():
        if bandera not in cohorte.columns:
            continue
        activa = cohorte[bandera].astype(str).str.lower().isin(["true", "1"])
        tasa = float(activa.mean())
        filas.append({
            "bandera": bandera,
            "tasa_base": round(tasa_base, 4),
            "tasa_cohorte": round(tasa, 4),
            "casos": int(activa.sum()),
            "razon": round(tasa / tasa_base, 2) if tasa_base > 0 else None,
        })
    return pd.DataFrame(filas)


def deriva_concepto(cohorte: pd.DataFrame, probabilidades: np.ndarray,
                    base: dict) -> dict:
    """Desempenio real contra el pago observado, con la salvedad de la madurez.

    Es la unica senial que confirma que el modelo se equivoca, y la unica que no
    se puede adelantar: exige que el credito venza.

    Reporta la cohorte COMPLETA y el subgrupo ya vencido por separado, en vez de
    filtrar y publicar un solo numero. El filtro de madurez parece el control
    limpio y no lo es: esta confundido con el plazo, porque un credito sigue vivo
    justamente por haberse pactado a mas meses. Sobre el test temporal, los
    vencidos promedian 6.0 meses de plazo y 2.09% de mora, y los jovenes 15.8
    meses y 4.17%. Publicar solo el subgrupo vencido reportaria el desempenio
    sobre la mitad mas corta y menos riesgosa de la cartera.

    El lift acompania siempre al AUC-PR porque las tasas base difieren entre
    subgrupos, y el AUC-PR no es comparable entre poblaciones con distinta
    proporcion de eventos.
    """
    if TARGET not in cohorte.columns:
        return {"medible": False,
                "motivo": "la cohorte no trae la variable objetivo"}

    import model_training as mt

    y = (1 - cohorte[TARGET]).to_numpy()

    def medir(mascara: np.ndarray) -> dict | None:
        if mascara.sum() < MIN_MUESTRA or y[mascara].sum() < 5:
            return None
        m = mt.summarize_classification(y[mascara], probabilidades[mascara],
                                        umbral=base["prediccion"]["umbral"],
                                        k=mt.K_OPERATIVO)
        return {"n": m["n"], "moras": m["eventos"], "tasa_mora": m["tasa_base"],
                "auc_pr": m["auc_pr"], "lift_vs_azar": m["lift_vs_azar"],
                "ks": m["ks"], "gini": m["gini"], "recall": m["recall"]}

    completo = medir(np.ones(len(y), dtype=bool))
    if completo is None:
        return {"medible": False,
                "motivo": (f"{len(y)} creditos con {int(y.sum())} moras, "
                           f"por debajo del minimo de {MIN_MUESTRA}")}

    resultado = {"medible": True, "cohorte_completa": completo}

    if "madurez_incompleta" in cohorte.columns:
        joven = cohorte["madurez_incompleta"].astype(str).str.lower().isin(
            ["true", "1"]).to_numpy()
        resultado["madurez_incompleta_pct"] = round(float(joven.mean()) * 100, 1)

        vencidos, jovenes = medir(~joven), medir(joven)
        if vencidos:
            vencidos["plazo_medio"] = round(
                float(cohorte.loc[~joven, "plazo_meses"].mean()), 1)
            resultado["solo_vencidos"] = vencidos
        if jovenes:
            jovenes["plazo_medio"] = round(
                float(cohorte.loc[joven, "plazo_meses"].mean()), 1)
            resultado["solo_jovenes"] = jovenes
        if vencidos and jovenes:
            resultado["sesgo_de_madurez"] = {
                "diferencia_plazo_meses": round(
                    jovenes["plazo_medio"] - vencidos["plazo_medio"], 1),
                "razon_mora": round(jovenes["tasa_mora"] / vencidos["tasa_mora"], 2)
                if vencidos["tasa_mora"] else None,
                "nota": ("El filtro de madurez selecciona creditos mas cortos, "
                         "no solo creditos vencidos. Ninguno de los dos subgrupos "
                         "da una lectura sin sesgo."),
            }

    return resultado


# ==============================================================================
# 5 | INFORME POR COHORTE
# ==============================================================================


def evaluar_cohorte(nombre: str, datos: pd.DataFrame, base: dict,
                    probabilidades: np.ndarray) -> dict:
    """Aplica las cuatro seniales a una cohorte y resume el veredicto.

    `probabilidades` es OBLIGATORIO, y no por comodidad. El monitoreo mide los
    pronosticos que el endpoint ENTREGO, no los que emitiria ahora: si el modelo
    se hubiera cambiado entre medias, recalcular borraria justo la diferencia que
    hay que detectar.

    La primera version lo aceptaba nulo y prometia calcularlo. Sobre datos del
    registro esa rama fallaba con un TypeError, porque la cadena espera
    `fecha_prestamo` como fecha y el registro la guarda como texto. Un parametro
    opcional que no funciona es peor que uno obligatorio.
    """
    if probabilidades is None or len(probabilidades) != len(datos):
        raise ValueError(
            f"'{nombre}': se esperaban {len(datos)} probabilidades y llegaron "
            f"{0 if probabilidades is None else len(probabilidades)}")

    preparada = preparar(datos)
    covariables = deriva_covariables(preparada, base)
    prediccion = deriva_prediccion(probabilidades, base)
    corte = estabilidad_del_corte(probabilidades, base)
    banderas = deriva_banderas(preparada, base)
    concepto = deriva_concepto(datos, probabilidades, base)

    alertadas = covariables[covariables["estado"] != ESTADO_ESTABLE]

    return {
        "cohorte": nombre,
        "n": int(len(datos)),
        "suficiente": bool(len(datos) >= MIN_MUESTRA),
        "covariables": {
            "psi_maximo": float(covariables["psi"].max()),
            "psi_medio": round(float(covariables["psi"].mean()), 4),
            "variables_alertadas": int(len(alertadas)),
            "tabla": covariables.to_dict(orient="records"),
        },
        "prediccion": prediccion,
        "corte": corte,
        "banderas": banderas.to_dict(orient="records"),
        "concepto": concepto,
        "veredicto": _veredicto(covariables, prediccion, corte),
    }


def _veredicto(covariables: pd.DataFrame, prediccion: dict,
               corte: dict) -> str:
    """Una linea que dice si hay que actuar, y no obliga a leer las tablas."""
    importantes = (covariables["estado"] == ESTADO_IMPORTANTE).sum()
    moderadas = (covariables["estado"] == ESTADO_MODERADO).sum()

    if importantes or prediccion["estado"] == ESTADO_IMPORTANTE:
        motivos = []
        if importantes:
            motivos.append(f"{importantes} variables con cambio importante")
        if prediccion["estado"] == ESTADO_IMPORTANTE:
            motivos.append(f"prediccion con PSI {prediccion['psi']}")
        return "REENTRENAR: " + " y ".join(motivos)
    if corte["requiere_revision"]:
        return (f"REVISAR EL CORTE: rechaza {corte['pct_rechazo_cohorte']}% "
                f"frente al {corte['pct_rechazo_base']}% previsto")
    if moderadas or prediccion["estado"] == ESTADO_MODERADO:
        return f"VIGILAR: {moderadas} variables con cambio moderado"
    return "SIN CAMBIO RELEVANTE"


def figura_deriva(informes: list[dict], destino: Path) -> Path:
    """Un grafico por variable, con las cohortes lado a lado.

    El control y la cohorte reciente en la misma figura: sin la referencia, una
    barra de PSI no dice si el numero es alto o si asi mide siempre.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / "psi_por_variable.png"

    tablas = {i["cohorte"]: pd.DataFrame(i["covariables"]["tabla"])
              for i in informes}
    orden = (tablas[informes[-1]["cohorte"]]
             .sort_values("psi", ascending=True)["variable"].tolist())

    fig, ax = plt.subplots(figsize=(9, 6))
    alto = 0.8 / len(tablas)
    posiciones = np.arange(len(orden))

    for i, (nombre, tabla) in enumerate(tablas.items()):
        valores = tabla.set_index("variable").reindex(orden)["psi"]
        ax.barh(posiciones + i * alto, valores, height=alto, label=nombre)

    for umbral, etiqueta in [(UMBRAL_PSI_MODERADO, "moderado 0.10"),
                             (UMBRAL_PSI_IMPORTANTE, "importante 0.25")]:
        ax.axvline(umbral, linestyle="--", linewidth=1, color="grey")
        ax.text(umbral, len(orden) - 0.3, etiqueta, fontsize=8,
                color="grey", ha="left")

    ax.set_yticks(posiciones + alto * (len(tablas) - 1) / 2)
    ax.set_yticklabels(orden, fontsize=9)
    ax.set_xlabel("PSI contra la linea base de entrenamiento")
    ax.set_title("Deriva de covariables por cohorte")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def figura_evolucion(periodos: dict, destino: Path) -> Path | None:
    """PSI maximo y volumen de rechazo a lo largo de las cosechas.

    Es la figura que responde la pregunta del enunciado: si la poblacion cambia
    de forma que afecte al modelo. Un PSI aislado no lo dice; la pendiente si.

    Las dos series van en el mismo eje temporal a proposito. Si la deriva
    estadistica y el volumen de rechazo suben juntos, lo que cambio son los
    clientes; si solo sube el rechazo, lo que fallo es el umbral.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    medibles = {k: v for k, v in periodos.items() if v.get("medible")}
    if len(medibles) < 3:
        return None

    etiquetas = list(medibles)
    psi_max = [v["covariables"]["psi_maximo"] for v in medibles.values()]
    rechazo = [v["corte"]["pct_rechazo_cohorte"] for v in medibles.values()]
    rechazo_base = list(medibles.values())[0]["corte"]["pct_rechazo_base"]

    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / "evolucion_por_cosecha.png"

    fig, (arriba, abajo) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    arriba.plot(etiquetas, psi_max, marker="o", color="#b03a2e")
    arriba.axhline(UMBRAL_PSI_MODERADO, linestyle="--", linewidth=1,
                   color="grey")
    arriba.axhline(UMBRAL_PSI_IMPORTANTE, linestyle="--", linewidth=1,
                   color="grey")
    arriba.text(0, UMBRAL_PSI_MODERADO, " moderado", fontsize=8, color="grey",
                va="bottom")
    arriba.text(0, UMBRAL_PSI_IMPORTANTE, " importante", fontsize=8,
                color="grey", va="bottom")
    arriba.set_ylabel("PSI maximo")
    arriba.set_title("Deriva por cosecha de desembolso")

    abajo.plot(etiquetas, rechazo, marker="o", color="#1f618d")
    abajo.axhline(rechazo_base, linestyle="--", linewidth=1, color="grey")
    abajo.text(0, rechazo_base, f" previsto {rechazo_base}%", fontsize=8,
               color="grey", va="bottom")
    abajo.set_ylabel("% rechazado")
    abajo.set_xlabel("Cosecha")
    abajo.tick_params(axis="x", rotation=45, labelsize=8)

    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


# ==============================================================================
# ORQUESTACION
# ==============================================================================


def _cohortes_de_contraste() -> dict:
    """Control y cohorte reciente, ambas con etiqueta para medir concepto.

    El control sale de la MISMA poblacion de ajuste, de modo que su PSI debe
    quedar cerca de cero por construccion. Es la prueba de que la medida
    funciona: sin ella, un PSI alto en la otra cohorte no se podria distinguir
    de un error de calculo.
    """
    train = pd.read_csv(RUTA_DATOS / "estratificado_train.csv",
                        sep=fe.SEPARADOR, encoding=fe.ENCODING)
    reciente = pd.read_csv(RUTA_DATOS / "temporal_test.csv",
                           sep=fe.SEPARADOR, encoding=fe.ENCODING)
    for datos in (train, reciente):
        datos["fecha_prestamo"] = pd.to_datetime(datos["fecha_prestamo"])

    return {
        "control (muestra de train)": train.sample(2000, random_state=SEMILLA),
        "reciente (temporal_test)": reciente,
    }


def origenes_registrados() -> set:
    """Etiquetas de origen que el registro ya contiene."""
    if not RUTA_REGISTRO.exists():
        return set()
    return set(cargar_registro()["origen"].unique())


def puntuar_cohorte(nombre: str, datos: pd.DataFrame,
                    ya_registrado: bool) -> tuple[np.ndarray, int]:
    """Puntua una cohorte por el endpoint, y la registra solo la primera vez.

    Se puntua LLAMANDO AL ENDPOINT y no invocando el modelo por atajo: es el
    mismo camino que recorre una solicitud real, de modo que lo medido es lo que
    el sistema hace y no una aproximacion.

    Pero registrar en cada corrida hacia crecer el registro sin limite con las
    mismas solicitudes: tras tres ejecuciones eran 12.562 filas para 3.801
    solicitudes distintas. Un area de riesgo no vuelve a puntuar el lote de ayer,
    y el monitoreo tampoco deberia fabricar el trafico que luego mide.
    """
    salida = md.predecir_lote(datos[md.COLUMNAS_REQUERIDAS],
                              origen="monitoreo:" + nombre,
                              guardar_registro=not ya_registrado)
    rechazos = int((salida["decision"] == md.DECISION_RECHAZAR).sum())
    return salida["probabilidad_mora"].to_numpy(dtype=float), rechazos


def main(reconstruir_base: bool = False) -> dict:
    log.info("=" * 72)
    log.info("FASE 5 | MONITOREO Y DATA DRIFT")
    log.info("=" * 72)

    # --- Linea base -----------------------------------------------------------
    log.info("PASO 1 | Linea base congelada desde el entrenamiento")
    base, construida = cargar_linea_base(reconstruir=reconstruir_base)
    log.info("   %s registros | %s variables | %s banderas",
             base["n"], len(base["variables"]), len(base["banderas"]))
    log.info("   probabilidad media %.5f | rechazo previsto %.2f%%",
             base["prediccion"]["probabilidad_media"],
             base["prediccion"]["pct_rechazo"])
    if construida:
        log.info("Guardado: %s", RUTA_LINEA_BASE.relative_to(RUTA_RAIZ))
    else:
        log.info("   leida de %s, no se reconstruye",
                 RUTA_LINEA_BASE.relative_to(RUTA_RAIZ))

    # --- Trafico por el endpoint --------------------------------------------
    log.info("-" * 72)
    log.info("PASO 2 | Las cohortes pasan por el endpoint")
    presentes = origenes_registrados()
    cohortes, probabilidades = {}, {}
    for nombre, datos in _cohortes_de_contraste().items():
        ya = ("monitoreo:" + nombre) in presentes
        probabilidades[nombre], rechazos = puntuar_cohorte(nombre, datos, ya)
        cohortes[nombre] = datos
        log.info("   %-28s %s solicitudes | %s rechazos | %s", nombre,
                 len(datos), rechazos,
                 "ya estaba en el registro" if ya else "anadida al registro")

    # --- Contraste con control ------------------------------------------------
    log.info("-" * 72)
    log.info("PASO 3 | Contraste: control contra cohorte reciente")
    informes = []
    for nombre, datos in cohortes.items():
        resultado = evaluar_cohorte(nombre, datos, base, probabilidades[nombre])
        informes.append(resultado)
        log.info("   %-28s n=%-5s PSI max %.4f (%s) | prediccion %.4f (%s)",
                 nombre, resultado["n"],
                 resultado["covariables"]["psi_maximo"],
                 clasificar(resultado["covariables"]["psi_maximo"]),
                 resultado["prediccion"]["psi"],
                 resultado["prediccion"]["estado"])
        log.info("   %-28s rechazo %.2f%% frente al %.2f%% previsto", "",
                 resultado["corte"]["pct_rechazo_cohorte"],
                 resultado["corte"]["pct_rechazo_base"])
        log.info("   %-28s %s", "", resultado["veredicto"])

    control, reciente = informes[0], informes[1]
    if control["covariables"]["psi_maximo"] >= UMBRAL_PSI_MODERADO:
        log.error("El control supera el umbral: la medida no es fiable.")
        raise RuntimeError("El control de la medicion de deriva fallo.")
    log.info("   control por debajo de %.2f: la medida discrimina",
             UMBRAL_PSI_MODERADO)

    # --- La tabla del endpoint, muestreada por periodo ------------------------
    log.info("-" * 72)
    log.info("PASO 4 | La tabla del endpoint, muestreada por periodo")
    registro = cargar_registro()
    log.info("   %s decisiones registradas | modelo %s", len(registro),
             ", ".join(registro["version_modelo"].unique()))

    informe_periodos = {}
    ejes = [("llegada", "momento", PERIODICIDAD, NOMBRE_PERIODICIDAD),
            ("cosecha", "fecha_prestamo", PERIODICIDAD_COSECHA,
             NOMBRE_PERIODICIDAD_COSECHA)]

    for eje, columna, frecuencia, nombre_eje in ejes:
        log.info("   eje %s (%s)", eje, nombre_eje)
        periodos = muestrear(registro, columna, frecuencia)
        publicados = 0
        informe_periodos[eje] = {"periodicidad": nombre_eje, "periodos": {}}

        for etiqueta, periodo in periodos.items():
            if not periodo["suficiente"]:
                informe_periodos[eje]["periodos"][etiqueta] = {
                    "n_decisiones": periodo["n_decisiones"],
                    "n_solicitudes": periodo["n_solicitudes"],
                    "medible": False,
                    "motivo": ("%s solicitudes distintas, bajo el minimo de %s"
                               % (periodo["n_solicitudes"], MIN_MUESTRA))}
                continue

            datos = periodo["datos"]
            prob = datos["probabilidad_mora"].to_numpy(dtype=float)
            validas = ~np.isnan(prob)
            resultado = evaluar_cohorte(eje + " " + etiqueta,
                                        datos.loc[validas], base, prob[validas])
            resultado["medible"] = True
            resultado["n_decisiones"] = periodo["n_decisiones"]
            resultado["n_solicitudes"] = periodo["n_solicitudes"]
            informe_periodos[eje]["periodos"][etiqueta] = resultado
            publicados += 1
            log.info("      %s  n=%-5s PSI max %.4f | rechazo %5.2f%%  %s",
                     etiqueta, periodo["n_solicitudes"],
                     resultado["covariables"]["psi_maximo"],
                     resultado["corte"]["pct_rechazo_cohorte"],
                     resultado["veredicto"])

        log.info("      %s periodos, %s publicados, %s bajo el minimo",
                 len(periodos), publicados, len(periodos) - publicados)

    # --- Las variables que mas se movieron ------------------------------------
    log.info("-" * 72)
    log.info("PASO 5 | Variables con mayor deriva en la cohorte reciente")
    for fila in reciente["covariables"]["tabla"][:5]:
        log.info("   %-34s PSI %.4f  %s", fila["variable"], fila["psi"],
                 fila["estado"])

    for fila in reciente["banderas"]:
        if fila["razon"] and fila["razon"] >= 1.5:
            log.warning("   bandera %s se activa %.2fx mas que en train "
                        "(%s casos)", fila["bandera"], fila["razon"],
                        fila["casos"])

    concepto = reciente["concepto"]
    if concepto["medible"]:
        subgrupos = [("cohorte_completa", "completa"),
                     ("solo_vencidos", "solo vencidos"),
                     ("solo_jovenes", "solo jovenes")]
        for clave, etiqueta in subgrupos:
            if clave in concepto:
                d = concepto[clave]
                plazo = ("" if "plazo_medio" not in d
                         else " | plazo %s meses" % d["plazo_medio"])
                log.info("   concepto %-14s n=%-5s mora %5.2f%% | AUC-PR %.4f "
                         "| lift %.2fx%s", etiqueta, d["n"],
                         d["tasa_mora"] * 100, d["auc_pr"],
                         d["lift_vs_azar"], plazo)
        sesgo = concepto.get("sesgo_de_madurez")
        if sesgo:
            log.warning("   el filtro de madurez esta confundido con el plazo: "
                        "%.1f meses de diferencia, mora %.2fx",
                        sesgo["diferencia_plazo_meses"], sesgo["razon_mora"])
    else:
        log.warning("   concepto no medible: %s", concepto["motivo"])

    # --- Figura y archivo -----------------------------------------------------
    log.info("-" * 72)
    figura = figura_deriva(informes, RUTA_FIGURAS)
    log.info("Guardado: %s", figura.relative_to(RUTA_RAIZ))

    evolucion = figura_evolucion(informe_periodos["cosecha"]["periodos"],
                                 RUTA_FIGURAS)
    if evolucion:
        log.info("Guardado: %s", evolucion.relative_to(RUTA_RAIZ))

    salida = {
        "periodicidad": {"llegada": NOMBRE_PERIODICIDAD,
                         "cosecha": NOMBRE_PERIODICIDAD_COSECHA},
        "umbrales_psi": base["umbrales_psi"],
        "linea_base": {"origen": base["origen"], "n": base["n"],
                       "version_modelo": base["version_modelo"]},
        "registro_endpoint": {
            "decisiones": int(len(registro)),
            "versiones_modelo": registro["version_modelo"].unique().tolist(),
            "minimo_por_periodo": MIN_MUESTRA,
            "nota_minimo": ("El minimo se aplica sobre SOLICITUDES DISTINTAS, no "
                            "sobre decisiones: una misma solicitud puntuada dos "
                            "veces es una observacion de la poblacion, no dos."),
        },
        "por_periodo": informe_periodos,
        "contraste": {i["cohorte"]: i for i in informes},
        "nota_concepto": ("La deriva de concepto se reporta sobre la cohorte "
                          "completa y sobre el subgrupo vencido por separado. El "
                          "filtro de madurez no aisla el censurado: esta "
                          "confundido con el plazo, de modo que selecciona "
                          "creditos mas cortos y menos riesgosos. Ninguno de los "
                          "dos subgrupos da una lectura sin sesgo."),
    }
    with open(RUTA_INFORME, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    log.info("Guardado: %s", RUTA_INFORME.relative_to(RUTA_RAIZ))
    log.info("=" * 72)

    return salida


if __name__ == "__main__":
    import argparse

    lector = argparse.ArgumentParser(
        description="Mide la deriva del modelo servido sobre el registro del "
                    "endpoint.")
    lector.add_argument(
        "--reconstruir-base", action="store_true",
        help="Rehace la linea base desde los datos de entrenamiento. Solo hace "
             "falta cuando cambia el modelo: la referencia pertenece a la "
             "version que la genero.")
    main(reconstruir_base=lector.parse_args().reconstruir_base)
