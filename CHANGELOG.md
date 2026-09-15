# Changelog

Qué trae cada versión entregada en `master`. Cada una tiene su etiqueta de Git y sus pull requests,
donde está el detalle y la verificación. El formato se basa en
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y las versiones siguen
[versionado semántico](https://semver.org/lang/es/).

## [2.0.1] - 2026-09-14

Incorpora la revisión de `danielCH26` sobre las PR6 y PR7. No cambia el modelo ni la imagen.

### Añadido

- `tests/test_evaluacion.py`: 13 tests sobre las funciones que tomaron las decisiones del modelo:
  la métrica, la regla de selección, la calibración y el umbral. De 6 errores introducidos a
  propósito en ellas, la suite detectó los 6.
- 3 tests de interacción entre reglas en `tests/test_despliegue.py`: varias anomalías en una misma
  solicitud y cuál domina.
- `fail_under = 40` en `.coveragerc`: si la cobertura baja de 40%, el CI falla.
- Este archivo y el procedimiento para volver a una versión anterior, en el readme.

### Cambiado

- La suite pasa de 182 a 198 tests y la cobertura de 43,0% a 44,3%.

### Sin cambios

- Modelo `0b9cf390cc9a`, umbral 0,0661 e imagen `mlops-riesgo:1.0.0`. Ninguno de los archivos que
  copia la imagen cambió, así que no hizo falta repetir la prueba de humo.

## [2.0.0] - 2026-09-13

Integra las PR #3 a #6. Es la primera versión con el modelo servido y monitoreado.

### Añadido

- **Evaluación (#3):** calibración, fairness, umbral de 0,0661 y scorecard de 45 filas, decididos
  sin tocar el test. El test se abrió una sola vez: ventaja sobre el piso heurístico de 1,78× en
  la partición estratificada y 1,45× en la temporal.
- **Despliegue (#4):** API FastAPI por lote con 5 endpoints, dentro de una imagen Docker con
  healthcheck. El saneamiento reproduce la limpieza de la Fase 1 con 0 diferencias en 10.763
  registros.
- **Monitoreo (#5):** PSI por cosecha sobre el registro que deja el endpoint, con grupo de control.
  El control da 0,0031; `plazo_meses` llega a 0,470 en diciembre de 2025.
- **Stage 2 (#6):** 182 tests con prueba de mutación (8 de 8 errores detectados), cobertura de
  43,0%, CI en GitHub Actions con SonarCloud y `model_monitoring.ipynb`.

### Cambiado

- `PRESENTACION.pptx` sale del control de versiones.

### Seguridad

- Las 7 alertas de *log injection* de SonarCloud se revisaron una a una y se marcaron como falso
  positivo: están en `main()`, que corre por consola, y registran cifras de los JSON del propio
  pipeline.

## [1.0.0] - 2026-09-09

Integra las PR #1 y #2.

### Añadido

- Estructura del Entregable 3 en `mlops_pipeline/`.
- `reglas_negocio.py` como contrato del EDA: bandas, cortes y reglas de validación.
- Feature Engineering cerrado: 17 características con WoE, sin fuga de información.
- Piso heurístico sobre `puntaje_datacredito` (AUC-PR 0,0753) y modelo seleccionado entre 9
  configuraciones: regresión logística sobre WoE, sin tratamiento del desbalance.

[2.0.1]: https://github.com/papiarcacamilo/MLOPS_CURSE/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/papiarcacamilo/MLOPS_CURSE/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/papiarcacamilo/MLOPS_CURSE/releases/tag/v1.0.0
