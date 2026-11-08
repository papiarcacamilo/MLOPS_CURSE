"""
Entregable 5, Fase 2 - Pipeline propio en Vertex AI Pipelines
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)

Reentrena el modelo final de Python, la logistica sobre WoE, como un pipeline
en la nube y lo mide igual que a AutoML:

    preparar_codigo -> cargar_datos -> entrenar -> evaluar -> [AUC-PR >= minimo] -> registrar

La logica de cada etapa esta en vertex_pasos.py. Aqui solo se declaran los
componentes, sus conexiones y la condicion para registrar, que es el mismo
minimo de AUC-PR que exige Jenkins en validar_metricas.py.

El codigo llega a la nube como un paquete del commit que se lanza, y cada etapa
instala las versiones fijadas en requirements.txt. Lanzar exige un arbol limpio
y un commit que ya este en GitHub: asi cada modelo se rastrea hasta su codigo.

Uso:
    python mlops_pipeline/src/vertex_pipeline.py             compila y muestra el plan
    python mlops_pipeline/src/vertex_pipeline.py --lanzar    sube el codigo y crea la corrida
================================================================================
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Sequence

from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

RAIZ = Path(__file__).resolve().parents[2]
RUTA_YAML = RAIZ / "infra" / "vertex" / "pipeline_mora_e5.yaml"
IMAGEN = "python:3.11-slim"

# Lo que viaja en el paquete: el codigo, la configuracion, el marcador con el
# que ft_engineering encuentra la raiz y la receta del modelo final.
ARCHIVOS_CODIGO = ("config.json", "Base_de_datos.csv", "mlops_pipeline/src/*.py",
                   "data/models/modelo_seleccionado.joblib")


def fijadas(paquetes: Sequence[str],
            archivos: Sequence[str] = ("requirements.txt", "requirements-vertex.txt")) -> list[str]:
    """Versiones de requirements, para que la nube instale lo mismo que el CI."""
    versiones = {}
    for archivo in archivos:
        for linea in (RAIZ / archivo).read_text(encoding="utf-8").splitlines():
            linea = linea.split("#")[0].strip()
            if "==" in linea:
                nombre, version = linea.split("==")
                versiones[nombre.strip().lower()] = version.strip()
    faltan = [p for p in paquetes if p.lower() not in versiones]
    if faltan:
        raise ValueError(f"sin version fijada en requirements: {faltan}")
    return [f"{p}=={versiones[p.lower()]}" for p in paquetes]


def auc_pr_minimo() -> float:
    """El mismo minimo que exige Jenkins, leido de su validador."""
    sys.path.insert(0, str(RAIZ / "ci"))
    from validar_metricas import AUC_PR_MINIMO
    return AUC_PR_MINIMO


PAQUETES = fijadas(["pandas", "numpy", "scipy", "scikit-learn", "joblib",
                    "imbalanced-learn", "sklearn-compat"])
PAQUETES_BIGQUERY = PAQUETES + fijadas(["pandas-gbq"])
# Kubeflow se instala en el contenedor sin sus dependencias, asi que la etapa
# que descarga el codigo declara el cliente de Cloud Storage.
PAQUETES_STORAGE = fijadas(["google-cloud-storage"])
PAQUETES_REGISTRO = PAQUETES + fijadas(["google-cloud-aiplatform"])


# ==============================================================================
# COMPONENTES
# ==============================================================================
# Kubeflow copia el cuerpo de cada funcion al contenedor, asi que cada una hace
# sus propios imports. Las que usan el repositorio lo agregan al path.


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES_STORAGE)
def preparar_codigo(codigo_uri: str, codigo: Output[Artifact]):
    """Descarga y extrae el paquete del commit que se lanzo."""
    import io
    import zipfile
    from pathlib import Path

    from google.cloud import storage

    bucket, _, nombre = codigo_uri.removeprefix("gs://").partition("/")
    datos = storage.Client().bucket(bucket).blob(nombre).download_as_bytes()
    zipfile.ZipFile(io.BytesIO(datos)).extractall(codigo.path)
    esenciales = ["config.json", "Base_de_datos.csv", "mlops_pipeline/src/vertex_pasos.py",
                  "data/models/modelo_seleccionado.joblib"]
    faltan = [f for f in esenciales if not (Path(codigo.path) / f).exists()]
    if faltan:
        raise FileNotFoundError(f"el paquete del codigo no trae: {faltan}")
    codigo.metadata["origen"] = codigo_uri


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES_BIGQUERY)
def cargar_datos(codigo: Input[Artifact], tabla: str, region: str,
                 train: Output[Dataset], test: Output[Dataset]):
    """Separa la tabla de BigQuery en train y test."""
    import sys
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(codigo.path) / "mlops_pipeline" / "src"))
    import vertex_pasos as vp

    conteos = vp.cargar_datos(tabla, region, train.path, test.path, raiz=Path(codigo.path))
    test.metadata.update(conteos)


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def entrenar(codigo: Input[Artifact], train: Input[Dataset], modelo: Output[Model]):
    """Ajusta la receta del modelo final con el train."""
    import sys
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(codigo.path) / "mlops_pipeline" / "src"))
    import vertex_pasos as vp

    modelo.metadata.update(vp.entrenar(train.path, modelo.path, raiz=Path(codigo.path)))
    modelo.metadata["framework"] = "scikit-learn"


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def evaluar(codigo: Input[Artifact], modelo: Input[Model], test: Input[Dataset],
            metricas: Output[Metrics],
            predicciones: Output[Dataset]) -> NamedTuple("Salida", [("auc_pr", float)]):
    """Mide el modelo sobre el test, rechazando el 20,9% de mayor riesgo."""
    import sys
    from collections import namedtuple
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(codigo.path) / "mlops_pipeline" / "src"))
    import vertex_pasos as vp

    m = vp.evaluar(modelo.path, test.path, predicciones.path, raiz=Path(codigo.path))
    for nombre in ("auc_pr", "roc_auc", "ks", "gini", "brier", "rmse",
                   "precision", "recall", "f1", "accuracy", "pct_marcado"):
        metricas.log_metric(nombre, float(m[nombre]))
    return namedtuple("Salida", ["auc_pr"])(float(m["auc_pr"]))


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES_REGISTRO)
def registrar(codigo: Input[Artifact], modelo: Input[Model], proyecto: str, region: str,
              nombre: str, commit: str) -> str:
    """Sube el modelo al Model Registry con la etiqueta del commit."""
    import sys
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(codigo.path) / "mlops_pipeline" / "src"))
    import vertex_pasos as vp

    etiquetas = {"entregable": "e5", "dominio": "riesgo", "uso": "comparacion-modelos",
                 "commit": commit[:12]}
    return vp.registrar(modelo.uri, proyecto, region, nombre, etiquetas)


@dsl.pipeline(name="mlops-mora-pipeline-e5",
              description="Reentrena la logistica sobre WoE y la mide como a AutoML.")
def pipeline(codigo_uri: str, tabla: str, proyecto: str, region: str, commit: str,
             nombre_modelo: str = "python-woe-pipeline-e5"):
    codigo = preparar_codigo(codigo_uri=codigo_uri).set_display_name("Preparar codigo")
    datos = cargar_datos(codigo=codigo.outputs["codigo"], tabla=tabla,
                         region=region).set_display_name("Cargar datos")
    entrenamiento = entrenar(codigo=codigo.outputs["codigo"],
                             train=datos.outputs["train"]).set_display_name("Entrenar")
    evaluacion = evaluar(codigo=codigo.outputs["codigo"],
                         modelo=entrenamiento.outputs["modelo"],
                         test=datos.outputs["test"]).set_display_name("Evaluar")
    with dsl.If(evaluacion.outputs["auc_pr"] >= auc_pr_minimo(), name="supera-el-minimo"):
        registrar(codigo=codigo.outputs["codigo"], modelo=entrenamiento.outputs["modelo"],
                  proyecto=proyecto, region=region, nombre=nombre_modelo,
                  commit=commit).set_display_name("Registrar")


def compilar(ruta: Path = RUTA_YAML) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=str(ruta))
    return ruta


# ==============================================================================
# LANZAMIENTO
# ==============================================================================


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit_lanzable(git=_git) -> str:
    """Devuelve el commit actual si se puede lanzar; si no, explica por que."""
    if git("status", "--porcelain"):
        raise RuntimeError("hay cambios sin commit: el modelo no se podria rastrear a su codigo")
    commit = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", commit):
        raise RuntimeError(f"el commit {commit[:7]} no esta en GitHub; haz push antes de lanzar")
    return commit


def empaquetar(commit: str, destino: Path, git=_git) -> Path:
    """Zip del commit con solo lo que el pipeline necesita."""
    git("archive", "--format=zip", "-o", str(destino), commit, "--", *ARCHIVOS_CODIGO)
    return destino


def subir_codigo(paquete: Path, bucket: str, commit: str, cliente=None) -> str:
    """Sube el paquete una vez por commit y devuelve su URI."""
    if cliente is None:
        from google.cloud import storage
        cliente = storage.Client()
    blob = cliente.bucket(bucket).blob(f"codigo/{commit}.zip")
    if not blob.exists():
        blob.upload_from_filename(str(paquete))
    return f"gs://{bucket}/codigo/{commit}.zip"


def plan(config: dict, commit: str, codigo_uri: str) -> dict:
    v, p = config["vertex"], config["vertex"]["pipeline"]
    return {
        "display_name": f"mlops-mora-pipeline-e5-{commit[:7]}",
        "pipeline_root": f"gs://{p['bucket']}/pipelines",
        "cuenta_servicio": p["cuenta_servicio"],
        "parametros": {
            "codigo_uri": codigo_uri,
            "tabla": f"{v['proyecto']}.{v['dataset']}.{v['tabla_pipeline']}",
            "proyecto": v["proyecto"],
            "region": v["region"],
            "commit": commit,
            "nombre_modelo": p["modelo"],
        },
        "etiquetas": {"entregable": "e5", "dominio": "riesgo",
                      "uso": "comparacion-modelos", "commit": commit[:12]},
    }


def lanzar(p: dict, plantilla: Path, aiplatform=None) -> str:
    """Crea la corrida sin esperar a que termine. Sin cache: cada corrida entrena."""
    if aiplatform is None:
        from google.cloud import aiplatform
    parametros = p["parametros"]
    aiplatform.init(project=parametros["proyecto"], location=parametros["region"])
    corrida = aiplatform.PipelineJob(
        display_name=p["display_name"], template_path=str(plantilla),
        pipeline_root=p["pipeline_root"], parameter_values=parametros,
        labels=p["etiquetas"], enable_caching=False)
    corrida.submit(service_account=p["cuenta_servicio"])
    return corrida.resource_name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--lanzar", action="store_true",
                        help="sube el codigo del commit y crea la corrida; sin esto solo compila")
    parser.add_argument("--salida", type=Path, default=RUTA_YAML,
                        help="donde escribir el pipeline compilado")
    args = parser.parse_args(argv)

    with open(RAIZ / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    plantilla = compilar(args.salida)
    print(f"Pipeline compilado: {plantilla}")

    if not args.lanzar:
        print(json.dumps(plan(config, "HEAD", "se sube al lanzar"), indent=2))
        print("Sin --lanzar no se sube ni se crea nada.")
        return 0

    commit = commit_lanzable()
    with tempfile.TemporaryDirectory() as carpeta:
        paquete = empaquetar(commit, Path(carpeta) / f"{commit}.zip")
        codigo_uri = subir_codigo(paquete, config["vertex"]["pipeline"]["bucket"], commit)
    p = plan(config, commit, codigo_uri)
    print(json.dumps(p, indent=2))
    print(f"Corrida creada: {lanzar(p, plantilla)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
