"""Pruebas de las compuertas de Jenkins (ci/).

Una compuerta que nunca rechaza no protege nada. Por eso, ademas de comprobar que
el repositorio real aprueba, cada regla se rompe a proposito con datos sinteticos
y debe rechazar sola. Los datos viven en carpetas temporales: estas pruebas no
tocan los artefactos del proyecto.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import validar_estructura as ve
import validar_metricas as vm

VERSION = "abc123def456"


def evaluacion_valida() -> dict:
    return {
        "umbral": {"umbral": 0.0661, "pct_rechazado": 20.91},
        "verificacion_scorecard": {"coincide": True},
        "evaluacion_final": {
            "estratificado": {
                "n_test": 2153,
                "eventos_test": 102,
                "metricas": {"auc_pr": 0.1479, "ks": 0.35, "brier": 0.04337},
                "calibracion": {"ece": 0.00846},
                "piso_en_el_mismo_test": {"auc_pr": 0.0829},
            },
            "temporal": {
                "n_test": 2154,
                "eventos_test": 69,
                "metricas": {"auc_pr": 0.0647, "ks": 0.35, "brier": 0.03},
                "calibracion": {"ece": 0.015},
                "piso_en_el_mismo_test": {"auc_pr": 0.0445},
            },
        },
    }


def artefacto_valido() -> dict:
    return {
        "modelo": {"version": VERSION},
        "umbral": {"valor": 0.0661},
        "desempenio_en_test": {"auc_pr": 0.1479},
    }


def poner(datos: dict, ruta: str, valor) -> None:
    """Cambia un valor anidado: poner(d, "a.b", 1) hace d["a"]["b"] = 1."""
    *padres, ultima = ruta.split(".")
    for clave in padres:
        datos = datos[clave]
    datos[ultima] = valor


def fallidos(evaluacion: dict, artefacto: dict, version: str = VERSION) -> list[str]:
    controles = vm.revisar_controles(evaluacion, artefacto, version)
    return [c.nombre for c in controles if not c.aprobado]


# ==============================================================================
# ESTRUCTURA
# ==============================================================================


@pytest.fixture
def repo_valido(tmp_path: Path) -> Path:
    for relativa in ve.REQUERIDOS:
        ruta = tmp_path / relativa
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("contenido", encoding="utf-8")
    return tmp_path


def test_estructura_completa_aprueba(repo_valido: Path) -> None:
    assert ve.revisar_estructura(repo_valido) == []
    assert ve.main(["--raiz", str(repo_valido)]) == 0


@pytest.mark.parametrize("relativa", ve.REQUERIDOS)
def test_falta_cualquier_archivo_rechaza(repo_valido: Path, relativa: str) -> None:
    (repo_valido / relativa).unlink()
    assert ve.revisar_estructura(repo_valido) == [f"falta {relativa}"]
    assert ve.main(["--raiz", str(repo_valido)]) == 1


def test_archivo_vacio_rechaza(repo_valido: Path) -> None:
    (repo_valido / "config.json").write_text("", encoding="utf-8")
    assert ve.revisar_estructura(repo_valido) == ["esta vacio config.json"]


def test_carpeta_en_lugar_de_archivo_rechaza(repo_valido: Path) -> None:
    (repo_valido / "readme.md").unlink()
    (repo_valido / "readme.md").mkdir()
    assert ve.revisar_estructura(repo_valido) == ["falta readme.md"]


def test_archivo_movido_fuera_de_src_rechaza(repo_valido: Path) -> None:
    origen = repo_valido / "mlops_pipeline" / "src" / "model_training.py"
    origen.rename(repo_valido / "model_training.py")
    assert ve.revisar_estructura(repo_valido) == [
        "falta mlops_pipeline/src/model_training.py"]


def test_estructura_real_del_repositorio_aprueba(raiz: Path) -> None:
    assert ve.revisar_estructura(raiz) == []


# ==============================================================================
# METRICAS: CONTROLES
# ==============================================================================


def test_metricas_validas_aprueban() -> None:
    assert fallidos(evaluacion_valida(), artefacto_valido()) == []


@pytest.mark.parametrize(
    ("ruta", "valor", "control"),
    [
        ("evaluacion_final.estratificado.n_test", 1999,
         "Tamanio del test estratificado"),
        ("evaluacion_final.temporal.n_test", 1999, "Tamanio del test temporal"),
        ("evaluacion_final.temporal.eventos_test", 49, "Tamanio del test temporal"),
        ("evaluacion_final.temporal.piso_en_el_mismo_test.auc_pr", 0.0600,
         "Supera al piso heuristico, temporal"),
        ("evaluacion_final.estratificado.piso_en_el_mismo_test.auc_pr", 0.1400,
         "Supera al piso heuristico, estratificado"),
        ("evaluacion_final.estratificado.metricas.brier", 0.05,
         "Brier mejor que el trivial, estratificado"),
        ("evaluacion_final.estratificado.calibracion.ece", 0.021,
         "ECE estratificado maximo"),
        ("umbral.pct_rechazado", 22.0, "Volumen de rechazo del umbral"),
        ("umbral.pct_rechazado", 19.8, "Volumen de rechazo del umbral"),
        ("verificacion_scorecard.coincide", False, "Scorecard equivalente al modelo"),
    ],
)
def test_cada_regla_de_evaluacion_rechaza_sola(ruta: str, valor, control: str) -> None:
    evaluacion = evaluacion_valida()
    poner(evaluacion, ruta, valor)
    assert fallidos(evaluacion, artefacto_valido()) == [control]


def test_pocas_moras_en_el_test_estratificado_rechaza() -> None:
    """Menos moras tambien bajan el Brier trivial: se ajusta para aislar la regla."""
    evaluacion = evaluacion_valida()
    poner(evaluacion, "evaluacion_final.estratificado.eventos_test", 49)
    poner(evaluacion, "evaluacion_final.estratificado.metricas.brier", 0.02)
    assert fallidos(evaluacion, artefacto_valido()) == ["Tamanio del test estratificado"]


def test_auc_pr_bajo_el_minimo_rechaza() -> None:
    evaluacion = evaluacion_valida()
    poner(evaluacion, "evaluacion_final.estratificado.metricas.auc_pr", 0.1319)
    assert "AUC-PR estratificado minimo" in fallidos(evaluacion, artefacto_valido())


def test_caida_dentro_del_ruido_no_rechaza() -> None:
    evaluacion = evaluacion_valida()
    artefacto = artefacto_valido()
    poner(evaluacion, "evaluacion_final.estratificado.metricas.auc_pr", vm.AUC_PR_MINIMO)
    poner(artefacto, "desempenio_en_test.auc_pr", vm.AUC_PR_MINIMO)
    assert fallidos(evaluacion, artefacto) == []


@pytest.mark.parametrize(
    ("ruta", "valor", "control"),
    [
        ("umbral.valor", 0.07, "Umbral publicado igual al evaluado"),
        ("desempenio_en_test.auc_pr", 0.15, "Desempenio publicado igual al evaluado"),
        ("modelo.version", "otra", "Version publicada igual al modelo guardado"),
    ],
)
def test_artefacto_desalineado_rechaza(ruta: str, valor, control: str) -> None:
    artefacto = artefacto_valido()
    poner(artefacto, ruta, valor)
    assert fallidos(evaluacion_valida(), artefacto) == [control]


def test_modelo_cambiado_sin_reevaluar_rechaza() -> None:
    assert fallidos(evaluacion_valida(), artefacto_valido(), "hash_distinto") == [
        "Version publicada igual al modelo guardado"]


@pytest.mark.parametrize("valor", [float("nan"), None, "0.15"])
def test_valor_invalido_rechaza_sin_romper(valor) -> None:
    evaluacion = evaluacion_valida()
    poner(evaluacion, "evaluacion_final.estratificado.calibracion.ece", valor)
    assert fallidos(evaluacion, artefacto_valido()) == ["ECE estratificado maximo"]


def test_scorecard_con_texto_no_cuenta_como_verdadero() -> None:
    evaluacion = evaluacion_valida()
    poner(evaluacion, "verificacion_scorecard.coincide", "true")
    assert fallidos(evaluacion, artefacto_valido()) == ["Scorecard equivalente al modelo"]


def test_particion_ausente_rechaza_sin_romper() -> None:
    evaluacion = evaluacion_valida()
    del evaluacion["evaluacion_final"]["temporal"]
    assert fallidos(evaluacion, artefacto_valido()) == [
        "Tamanio del test temporal", "Supera al piso heuristico, temporal"]
    assert vm.revisar_avisos(evaluacion) == []


# ==============================================================================
# METRICAS: AVISOS
# ==============================================================================


def test_sin_limitaciones_no_hay_avisos() -> None:
    assert vm.revisar_avisos(evaluacion_valida()) == []


def test_limitaciones_avisan_pero_no_rechazan() -> None:
    evaluacion = evaluacion_valida()
    poner(evaluacion, "evaluacion_final.estratificado.metricas.ks", 0.2742)
    poner(evaluacion, "evaluacion_final.temporal.metricas.ks", 0.2026)
    poner(evaluacion, "evaluacion_final.temporal.metricas.brier", 0.03332)
    poner(evaluacion, "evaluacion_final.temporal.calibracion.ece", 0.03137)

    avisos = " | ".join(vm.revisar_avisos(evaluacion))
    assert "KS estratificado" in avisos
    assert "KS temporal" in avisos
    assert "Brier temporal" in avisos
    assert "ECE temporal" in avisos
    assert fallidos(evaluacion, artefacto_valido()) == []


# ==============================================================================
# METRICAS: PUNTO DE ENTRADA
# ==============================================================================


@pytest.fixture
def repo_con_artefactos(tmp_path: Path) -> Path:
    modelos = tmp_path / "data" / "models"
    modelos.mkdir(parents=True)
    contenido = b"modelo sintetico"
    (modelos / "modelo_seleccionado.joblib").write_bytes(contenido)

    artefacto = artefacto_valido()
    artefacto["modelo"]["version"] = hashlib.sha256(contenido).hexdigest()[:12]
    (modelos / "artefacto_despliegue.json").write_text(
        json.dumps(artefacto), encoding="utf-8")
    (modelos / "evaluacion.json").write_text(
        json.dumps(evaluacion_valida()), encoding="utf-8")
    return tmp_path


def test_main_aprueba_con_artefactos_validos(repo_con_artefactos: Path, capsys) -> None:
    assert vm.main(["--raiz", str(repo_con_artefactos)]) == 0
    assert "APROBADO" in capsys.readouterr().out


def test_main_rechaza_si_cambia_el_modelo(repo_con_artefactos: Path, capsys) -> None:
    ruta = repo_con_artefactos / vm.RUTA_MODELO
    ruta.write_bytes(b"otro modelo")
    assert vm.main(["--raiz", str(repo_con_artefactos)]) == 1
    salida = capsys.readouterr().out
    assert "[FALLA] Version publicada" in salida
    assert "RECHAZADO: 1 de 12" in salida


@pytest.mark.parametrize("relativa", [vm.RUTA_EVALUACION, vm.RUTA_ARTEFACTO, vm.RUTA_MODELO])
def test_main_rechaza_si_falta_un_artefacto(repo_con_artefactos: Path, relativa: str,
                                            capsys) -> None:
    (repo_con_artefactos / relativa).unlink()
    assert vm.main(["--raiz", str(repo_con_artefactos)]) == 1
    assert "RECHAZADO" in capsys.readouterr().out


def test_main_rechaza_json_corrupto(repo_con_artefactos: Path, capsys) -> None:
    (repo_con_artefactos / vm.RUTA_EVALUACION).write_text("{ no es json", encoding="utf-8")
    assert vm.main(["--raiz", str(repo_con_artefactos)]) == 1
    assert "RECHAZADO" in capsys.readouterr().out


def test_metricas_reales_del_repositorio_aprueban(raiz: Path, capsys) -> None:
    assert vm.main(["--raiz", str(raiz)]) == 0
    salida = capsys.readouterr().out
    assert "APROBADO" in salida
    assert "[AVISO] KS" in salida
