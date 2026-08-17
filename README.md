# MLOPS_CURSE — Ciencia de Datos en Producción

## Descripción del proyecto

Proyecto de análisis de riesgo crediticio para una empresa financiera. A partir de una base de datos de créditos con información del cliente, del crédito y del comportamiento de pago (a tiempo o en mora), se desarrolla un pipeline completo de MLOps: desde la exploración y limpieza de datos hasta el despliegue y monitoreo de un modelo predictivo.

## Estructura del repositorio

```
MLOPS_CURSE/
├── src/
│   ├── transformacion_eda.ipynb      # EDA + limpieza + insights (Fase 0 y 1)
│   ├── config.json                   # Configuración del proyecto
│   ├── ft_engineering.py             # Feature Engineering (Fase 2)
│   ├── model_training_evaluation.py  # Entrenamiento y evaluación (Fase 3)
│   ├── model_deploy.py               # Despliegue del modelo (Fase 4)
│   └── model_monitoring.py           # Monitoreo en producción (Fase 5)
├── Base_de_datos.csv                 # Datos crudos originales (no modificar)
├── Base_de_datos_limpia.csv          # Datos limpios (generado por el notebook)
├── requirements.txt                  # Dependencias del proyecto
├── setup.bat                         # Script de configuración inicial
├── .gitignore                        # Exclusiones de Git
└── README.md                         # Este archivo
```

## Ramas del repositorio

| Rama | Propósito |
|---|---|
| `developer` | Rama de trabajo activo (commits diarios, experimentación) |
| `master` | Rama estable (merges al cerrar cada fase validada) |
| `certification` | Rama de certificación / entregable final |

## Dataset

- **Fuente**: `Base_de_datos.csv` (10.763 registros × 23 columnas)
- **Variable objetivo**: `Pago_atiempo` (1 = pagó a tiempo, 0 = cayó en mora)
- **Desbalance**: 95.25% clase mayoritaria (al día), 4.75% clase minoritaria (mora)
- **No se proporcionó diccionario de datos**: la interpretación de las variables se realizó con criterio de negocio financiero e investigación del contexto colombiano (Datacrédito)

## Diccionario de datos (inferido)

| Variable | Tipo | Descripción |
|---|---|---|
| `tipo_credito` | Categórica nominal | Código de línea/producto de crédito (4, 6, 7, 9, 10, 68) |
| `fecha_prestamo` | Fecha | Fecha y hora de desembolso del crédito |
| `capital_prestado` | Numérica continua | Monto del crédito otorgado (COP) |
| `plazo_meses` | Numérica discreta | Plazo del crédito en meses |
| `edad_cliente` | Numérica discreta | Edad del cliente al momento del crédito |
| `tipo_laboral` | Categórica dicotómica | Situación laboral (Empleado / Independiente) |
| `salario_cliente` | Numérica continua | Ingreso mensual reportado (COP) |
| `total_otros_prestamos` | Numérica continua | Deuda en otros créditos (COP) |
| `cuota_pactada` | Numérica continua | Cuota mensual del crédito (COP) |
| `puntaje_datacredito` | Numérica discreta | Score de central de riesgo (Datacrédito) |
| `cant_creditosvigentes` | Numérica discreta | Cantidad de créditos activos del cliente |
| `huella_consulta` | Numérica discreta | Consultas recientes a la central de riesgo |
| `saldo_mora` | Numérica continua | Saldo actualmente en mora (COP) |
| `saldo_total` | Numérica continua | Saldo total pendiente (COP) |
| `saldo_principal` | Numérica continua | Saldo de capital pendiente (COP) |
| `saldo_mora_codeudor` | Numérica continua | Mora atribuible a codeudor (COP) |
| `creditos_sectorFinanciero` | Numérica discreta | Créditos en el sector financiero |
| `creditos_sectorCooperativo` | Numérica discreta | Créditos en el sector cooperativo |
| `creditos_sectorReal` | Numérica discreta | Créditos en el sector real |
| `promedio_ingresos_datacredito` | Numérica continua | Ingreso promedio reportado en central de riesgo |
| `tendencia_ingresos` | Categórica politómica | Tendencia de ingresos (Creciente / Decreciente / Estable) |
| `Pago_atiempo` | Categórica dicotómica | **Variable objetivo** (1 = al día, 0 = mora) |

## Principales hallazgos

1. **Mora previa es la señal más potente**: clientes con saldo en mora tienen 36.36% de tasa de incumplimiento vs. 4.59% sin mora previa (~8x más riesgo)
2. **Tipo de crédito 6 concentra riesgo**: tasa de mora de 42.86% vs. ~4.75% promedio
3. **No hay un predictor lineal fuerte individual**: ninguna variable original supera |r| = 0.08; el valor está en la combinación de variables y atributos derivados
4. **Los atributos derivados superan a las originales**: `tiene_mora_previa` y `consultas_por_credito` lideran el ranking de correlación con la variable objetivo

## Problemas de calidad detectados y tratados

- `puntaje`: eliminada (87.4% de valores idénticos, sin capacidad predictiva)
- `edad_cliente`: 150 registros con 121-123 años → imputados con mediana (42)
- `salario_cliente`: 250 registros extremos → corregidos con IQR sobre escala logarítmica
- `tendencia_ingresos`: 58 registros con valores numéricos corruptos → reconstruidos por signo
- Nulos en saldos: imputados con 0 (causa estructural: ausencia de otros créditos)
- Nulos en `promedio_ingresos_datacredito`: dejados como NaN intencionalmente (27%, sin causa estructural)

## Instalación y ejecución

```bash
# Clonar el repositorio
git clone https://github.com/papiarcacamilo/MLOPS_CURSE.git
cd MLOPS_CURSE

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el notebook de EDA y limpieza
jupyter notebook src/transformacion_eda.ipynb
```

## Tecnologías utilizadas

- Python 3.11
- pandas, numpy (manipulación de datos)
- matplotlib, seaborn (visualización)
- scipy (estadística)
- jupyter (notebooks)

## Autor

Camilo Andrés Armenta — [papiarcacamilo](https://github.com/papiarcacamilo)

## Docente

Juan Sebastián Parra Sánchez — Ciencia de Datos en Producción
