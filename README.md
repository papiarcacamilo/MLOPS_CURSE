# MLOPS_CURSE — Análisis de Riesgo Crediticio

Proyecto transversal de **Ciencia de Datos en Producción**. Construye un pipeline MLOps completo
sobre una base de datos real de créditos de una empresa financiera colombiana, desde la
comprensión y limpieza de los datos hasta el despliegue y monitoreo de un modelo predictivo.

> **Estado actual: Fase 1 completada** (EDA, limpieza y diccionario de datos).
> Las fases de Feature Engineering, Modelado, Despliegue y Monitoreo están pendientes.
> Este README describe únicamente lo que el código implementa hoy.

---

## Tabla de contenido

1. [Problema planteado](#problema-planteado)
2. [Objetivos](#objetivos)
3. [Contexto](#contexto)
4. [Dataset](#dataset)
5. [Diccionario de datos](#diccionario-de-datos)
6. [Categorización de variables](#categorización-de-variables)
7. [Metodología](#metodología)
8. [Tratamiento y limpieza de datos](#tratamiento-y-limpieza-de-datos)
9. [Análisis exploratorio](#análisis-exploratorio)
10. [Reglas de validación](#reglas-de-validación)
11. [Resultados](#resultados)
12. [Conclusiones](#conclusiones)
13. [Estructura del repositorio](#estructura-del-repositorio)
14. [Tecnologías utilizadas](#tecnologías-utilizadas)
15. [Instrucciones de ejecución](#instrucciones-de-ejecución)
16. [Referencias](#referencias)

---

## Problema planteado

Una empresa financiera entrega una base de datos con información del crédito, del cliente y del
resultado de pago (a tiempo o en mora). Se requiere **generar insights valiosos para la
organización y comunicarlos adecuadamente**, con el fin de apoyar decisiones de riesgo
crediticio.

El reto adicional: **no se proporcionó diccionario de datos**, por lo que la interpretación y
categorización de cada variable debió construirse mediante investigación del contexto bancario
colombiano.

## Objetivos

### Objetivo general

Analizar una base de datos de créditos para identificar los factores asociados al incumplimiento
de pago, y construir la base documentada y depurada sobre la que se desarrollará un modelo
predictivo de riesgo crediticio.

### Objetivos específicos

1. Construir un diccionario de datos completo y técnicamente fundamentado, en ausencia de
   documentación original.
2. Evaluar la calidad de los datos e identificar errores de captura, valores atípicos y
   registros inconsistentes.
3. Aplicar un proceso de limpieza y transformación trazable y reproducible.
4. Realizar análisis exploratorio univariado, bivariado y multivariado.
5. Definir reglas de validación reutilizables en etapas posteriores del pipeline.
6. Identificar atributos derivados con potencial predictivo.
7. Comunicar los hallazgos de forma clara para perfiles técnicos y de negocio.

## Contexto

El análisis se sitúa en el sistema financiero colombiano. Las variables provenientes de central
de riesgo corresponden al modelo de **DataCrédito Experian**, cuyo score crediticio opera en un
rango oficial de **150 a 950 puntos**, donde un valor más alto indica menor riesgo de
incumplimiento.

En cuanto a protección de datos, la **Ley 1266 de 2008** (Habeas Data financiero) clasifica el
dato financiero y crediticio como **semiprivado**. La **Ley 1581 de 2012 (art. 5)** reserva la
categoría de dato **sensible** para información como origen racial, orientación política,
convicciones religiosas, salud o biometría — ninguna variable de este dataset entra en esa
categoría.

## Dataset

| Atributo | Valor |
|---|---|
| Archivo fuente | `Base_de_datos.csv` |
| Registros | 10.763 créditos |
| Variables originales | 23 |
| Variables tras limpieza | 31 (incluye columnas de trazabilidad) |
| Separador | `;` |
| Codificación | UTF-8 con BOM |
| Rango temporal | 2024-11-26 a 2026-04-26 |
| Variable objetivo | `Pago_atiempo` (1 = al día, 0 = mora) |
| Distribución del objetivo | 95.25% al día / 4.75% mora (ratio 20:1) |

**Fuente de los datos:** base de datos entregada por el docente del curso, correspondiente a
operaciones de crédito de una entidad financiera. No se proporcionó documentación adjunta.

**Unidad de observación:** el crédito, no el cliente. El dataset **no contiene identificador de
cliente** (ver [Limitaciones](#limitaciones)).

## Diccionario de datos

El diccionario está implementado como estructura de código en `src/transformacion_eda.ipynb`
(constante `DICCIONARIO_DATOS`), de la cual el resto del análisis deriva sus listas de variables.
Esto evita inferir el tipo estadístico desde el tipo de dato técnico.

| Variable | Descripción | Tipo dato | Tipo variable | Subtipo | Nivel medición | Unidad | Valores / rango | Sensibilidad | Rol |
|---|---|---|---|---|---|---|---|---|---|
| `tipo_credito` | Código interno de la línea de crédito | int64 | Cualitativa | Nominal politómica | Nominal | — | 4, 6, 7, 9, 10, 68 | Semiprivado | Predictor categórico |
| `fecha_prestamo` | Fecha y hora de desembolso | datetime64 | Cuantitativa | Temporal | Intervalo | dd/mm/aaaa hh:mm | 2024-11-26 a 2026-04-26 | Semiprivado | Fuente de derivadas |
| `capital_prestado` | Monto de capital desembolsado | int64 | Cuantitativa | Continua | Razón | COP | 360.000 – 41.444.150 | Semiprivado | Predictor |
| `plazo_meses` | Plazo pactado | int64 | Cuantitativa | Discreta | Razón | meses | 2 – 90 | Semiprivado | Predictor |
| `edad_cliente` | Edad del cliente | int64 | Cuantitativa | Discreta | Razón | años | 19 – 69 (tras limpieza) | Semiprivado | Predictor |
| `tipo_laboral` | Situación laboral | object | Cualitativa | Nominal dicotómica | Nominal | — | Empleado, Independiente | Semiprivado | Predictor categórico |
| `salario_cliente` | Ingreso mensual declarado | int64 | Cuantitativa | Continua | Razón | COP/mes | 1.000 – 18.756.570 (tras limpieza) | Semiprivado | Predictor |
| `total_otros_prestamos` | Otras obligaciones del cliente | int64 | Cuantitativa | Continua | Razón | COP | ≥ 0 | Semiprivado | Predictor |
| `cuota_pactada` | Cuota mensual acordada | int64 | Cuantitativa | Continua | Razón | COP/mes | ≥ 23.944 | Semiprivado | Predictor |
| `puntaje_datacredito` | Score de central de riesgo | float64 | Cuantitativa | Discreta | Intervalo | puntos | **150 – 950** | Semiprivado | Predictor |
| `cant_creditosvigentes` | Créditos activos del cliente | int64 | Cuantitativa | Discreta (conteo) | Razón | créditos | 0 – 62 | Semiprivado | Predictor |
| `huella_consulta` | Consultas recientes en central de riesgo | int64 | Cuantitativa | Discreta (conteo) | Razón | consultas | 0 – 30 | Semiprivado | Predictor |
| `saldo_mora` | Saldo en mora del cliente | float64 | Cuantitativa | Continua | Razón | COP | ≥ 0 | Semiprivado | Predictor ⚠️ fuga |
| `saldo_total` | Saldo total pendiente | float64 | Cuantitativa | Continua | Razón | COP | ≥ 0 | Semiprivado | Predictor |
| `saldo_principal` | Saldo de capital pendiente | float64 | Cuantitativa | Continua | Razón | COP | ≥ 0 | Semiprivado | Predictor |
| `saldo_mora_codeudor` | Mora atribuible al codeudor | float64 | Cuantitativa | Continua | Razón | COP | ≥ 0 | Semiprivado | Predictor ⚠️ fuga |
| `creditos_sectorFinanciero` | Créditos en sector financiero | int64 | Cuantitativa | Discreta (conteo) | Razón | créditos | 0 – 51 | Semiprivado | Predictor |
| `creditos_sectorCooperativo` | Créditos en sector cooperativo | int64 | Cuantitativa | Discreta (conteo) | Razón | créditos | 0 – 13 | Semiprivado | Predictor |
| `creditos_sectorReal` | Créditos en sector real | int64 | Cuantitativa | Discreta (conteo) | Razón | créditos | 0 – 25 | Semiprivado | Predictor |
| `promedio_ingresos_datacredito` | Ingreso promedio según central de riesgo | float64 | Cuantitativa | Continua | Razón | COP/mes | ≥ 0 | Semiprivado | Predictor (27% nulos) |
| `tendencia_ingresos` | Tendencia de ingresos | object | Cualitativa | Nominal politómica | Nominal | — | Creciente, Decreciente, Estable, Sin_dato | Semiprivado | Predictor categórico |
| `Pago_atiempo` | Resultado de pago del crédito | int64 | Cualitativa | Nominal dicotómica | Nominal | — | 0, 1 | Semiprivado | **Variable objetivo** |

### Notas críticas del diccionario

- **`tipo_credito`**: los códigos 4, 6, 7, 9, 10 y 68 **no corresponden a ninguna clasificación
  oficial de la Superintendencia Financiera** (que usa modalidades: comercial, consumo, vivienda,
  microcrédito). Se documentan como códigos internos de significado no confirmado.
- **`puntaje_datacredito`**: el valor `0` **no es un score**. DataCrédito confirma que no existen
  puntajes nulos ni negativos. Los 145 registros con `0` se interpretan como **ausencia de
  historial crediticio** y reciben tratamiento separado.
- **`saldo_mora` y `saldo_mora_codeudor`**: marcadas con riesgo de **fuga de información**. Debe
  confirmarse con el negocio si se registran antes del desembolso.

### Columnas de trazabilidad generadas

El proceso de limpieza añade columnas booleanas que documentan qué registros fueron modificados,
permitiendo auditar cada intervención:

`edad_cliente_corregida`, `salario_cliente_corregido`, `tendencia_ingresos_reconstruida`,
`sin_historial_crediticio`, `saldo_principal_era_nulo`, `saldo_mora_era_nulo`,
`saldo_total_era_nulo`, `saldo_mora_codeudor_era_nulo`, `promedio_ingresos_datacredito_era_nulo`.

## Categorización de variables

| Categoría | Variables |
|---|---|
| **Cualitativas nominales** (4) | `tipo_credito`, `tipo_laboral`, `tendencia_ingresos`, `Pago_atiempo` |
| **Cuantitativas continuas** (9) | `capital_prestado`, `salario_cliente`, `total_otros_prestamos`, `cuota_pactada`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`, `promedio_ingresos_datacredito` |
| **Cuantitativas discretas** (8) | `plazo_meses`, `edad_cliente`, `puntaje_datacredito`, `cant_creditosvigentes`, `huella_consulta`, `creditos_sectorFinanciero`, `creditos_sectorCooperativo`, `creditos_sectorReal` |
| **Temporales** (1) | `fecha_prestamo` |

Agrupación funcional: **Producto** (tipo de crédito) · **Demográfica** (edad, situación laboral)
· **Capacidad de pago** (salario) · **Condiciones del crédito** (capital, plazo, cuota) ·
**Buró de crédito** (score, consultas, créditos por sector, ingresos reportados) ·
**Comportamiento de pago** (saldos en mora) · **Resultado** (variable objetivo).

## Metodología

```
1. Carga de datos
2. Exploración inicial      → diccionario de datos, caracterización, nulos
3. Limpieza y corrección    → tipos, atípicos, valores inválidos
4. EDA                      → univariado → bivariado → multivariado
5. Reglas de validación     → dominio válido por variable, función reutilizable
6. Identificación del target
7. Transformaciones futuras y atributos derivados
8. Conclusiones e insights
9. Guardado del dataset limpio
```

El principio rector: **entender antes de limpiar, limpiar antes de analizar**. Cada decisión de
limpieza se investigó empíricamente antes de aplicarse, y quedó documentada con su justificación.

## Tratamiento y limpieza de datos

| Problema detectado | Registros | Tratamiento aplicado | Justificación |
|---|---|---|---|
| `puntaje` con 87.4% de valores idénticos | 9.407 | Columna eliminada | Sin capacidad discriminante; se comporta como constante |
| `fecha_prestamo` como texto | 10.763 | Conversión a `datetime` | Permite análisis temporal |
| Edades imposibles (121–123 años) | 150 | Imputación con mediana (42) | Salto claro en la distribución: sin registros entre 70 y 120 |
| Salarios extremos y en cero | 250 | Imputación con mediana ($3M) | IQR sobre escala logarítmica (técnica para variables monetarias asimétricas) |
| `tendencia_ingresos` con valores numéricos | 58 | Reconstrucción por signo | Los 58 tenían `promedio_ingresos_datacredito` poblado: fallo de categorización en origen |
| **Score fuera del rango [150, 950]** | **153** | **Marcado como `sin_historial_crediticio`, score → NaN** | **No son scores bajos: son ausencia de score (fuente: DataCrédito Experian)** |
| Nulos en saldos | 405 / 590 | Imputación con 0 | Causa estructural confirmada: `cant_creditosvigentes = 0` |
| Nulos en `puntaje_datacredito` | 6 | Imputación con mediana | Impacto insignificante (0.06%) |
| Nulos en `promedio_ingresos_datacredito` | 2.930 | **No imputados** (NaN + bandera) | 27% sin causa estructural: imputar fabricaría el dato |
| Nulos en `tendencia_ingresos` | 2.932 | Categoría explícita `Sin_dato` | Preserva la información de ausencia |

**Salida:** `Base_de_datos_limpia.csv` (10.763 × 31). El archivo crudo `Base_de_datos.csv`
permanece intacto para garantizar reproducibilidad desde cero.

## Análisis exploratorio

### Análisis univariado

- **Cuantitativas:** `describe()` completo, medidas de dispersión (rango, IQR, varianza,
  desviación estándar, **skewness**, **kurtosis**), histogramas de 11 variables clave y boxplots
  de edad y salario.
- **Cualitativas:** tablas de frecuencia absoluta y relativa, countplots de las 4 variables
  nominales.

**Hallazgos de forma de distribución:**

| Variable | Skewness | Kurtosis | Distribución |
|---|---|---|---|
| `edad_cliente` | 0.27 | −0.83 | Aproximadamente simétrica / cuasi-gaussiana |
| `capital_prestado` | 3.72 | 35.3 | Fuertemente sesgada a la derecha (log-normal) |
| `salario_cliente` | 2.20 | 6.34 | Sesgada a la derecha (log-normal) |
| `puntaje_datacredito` | −5.65 | 39.4 | Sesgada a la izquierda y **bimodal** |
| `saldo_mora_codeudor` | 97.7 | 9.813 | Extremadamente sesgada (mayoría en 0) |

### Análisis bivariado

- **Cuantitativa vs. objetivo:** comparación de medianas por grupo y boxplots comparativos de 6
  variables clave.
- **Cualitativa vs. objetivo:** tasas de mora por categoría (barplots) y **prueba chi-cuadrado
  de independencia con V de Cramér** — técnica correcta para variables nominales, en lugar de
  correlación de Pearson.

**Resultados de la prueba chi-cuadrado:**

| Variable | χ² | gl | p-valor | V de Cramér | ¿Asociación significativa? |
|---|---|---|---|---|---|
| `tipo_credito` | 68.86 | 5 | 1.77e−13 | 0.0800 | Sí |
| `tendencia_ingresos` | 21.48 | 3 | 8.38e−05 | 0.0447 | Sí |
| `tipo_laboral` | 8.00 | 1 | 0.0047 | 0.0273 | Sí |

### Análisis multivariado

- **Matriz de correlación** sobre variables cuantitativas (excluye nominales por definición del
  diccionario) + detección automática de multicolinealidad.
- **Pairplot** con `hue` en la variable objetivo (muestra de 1.500 registros).
- **Tablas de contingencia** cruzadas entre variables categóricas.

**Correlación con la variable objetivo (top 5):**

| Variable | \|r\| |
|---|---|
| `puntaje_datacredito` | **0.1212** |
| `huella_consulta` | 0.0737 |
| `saldo_mora` | 0.0726 |
| `plazo_meses` | 0.0631 |
| `edad_cliente` | 0.0524 |

**Multicolinealidad:** `capital_prestado` ↔ `cuota_pactada` (r = 0.764).


## Reglas de validación

Implementadas como diccionario `REGLAS_VALIDACION` y función `validar_dataframe()`, reutilizables
en fases posteriores del pipeline y para validar datos nuevos en producción. Cada regla incluye
su campo `fuente`.

**Corrección relevante:** la regla de `puntaje_datacredito` era `[0, 1000]` — un rango sin
fuente. Corregida a `[150, 950]` (rango oficial DataCrédito Experian). La regla anterior detectaba
**1** registro inválido; la corregida detecta **153**.

## Ingeniería de características

>  Corresponde a la Fase 2
> (`src/ft_engineering.py`).

Se propusieron y evaluaron 9 atributos derivados dentro del notebook, midiendo su correlación
real con la variable objetivo:

| Atributo derivado | \|r\| | Evaluación |
|---|---|---|
| `tiene_mora_previa` | 0.107 | **Mejor predictor derivado** (36.36% vs 4.59% de mora) |
| `consultas_por_credito` | 0.077 | Supera a `huella_consulta` original (0.074) |
| `ratio_deuda_ingreso` | 0.007 | Descartable: señal casi nula |
| `ratio_cuota_ingreso` | 0.003 | Descartable: señal casi nula |
| `ratio_capital_ingreso` | 0.0002 | Descartable: señal casi nula |

Los ratios clásicos de capacidad de pago no funcionaron en este dataset, probablemente
porque 250 salarios fueron imputados con la mediana durante la limpieza, distorsionando
cualquier cociente calculado sobre esa base.

Transformaciones planificadas: One-Hot Encoding
para categóricas, transformación logarítmica para variables monetarias asimétricas, `log1p` para
variables con ceros, escalado robusto, y extracción de componentes temporales.



## Resultados

### Insights principales

1. **La mora previa multiplica el riesgo por 8.** Clientes con saldo en mora registrado: 36.36%
   de incumplimiento vs. 4.59% sin ella (n = 55).
2. **La ausencia de historial crediticio también es señal de riesgo.** Los 153 clientes sin score
   válido presentan 6.54% de mora vs. 4.72% de quienes sí lo tienen.
3. **El tipo de crédito 6 concentra riesgo desproporcionado:** 42.86% de mora vs. 4.75% promedio
   (n = 21). Asociación confirmada por chi-cuadrado (p = 1.77e−13).
4. **Depurar el score mejoró su poder predictivo un 78%:** de |r| = 0.068 a |r| = 0.1212, pasando
   a ser el predictor cuantitativo más fuerte.
5. **Los independientes presentan mayor riesgo** que los empleados: 5.51% vs. 4.29%.
6. **Los atributos derivados superan a la mayoría de variables originales** en poder predictivo.

## Conclusiones

- La calidad de los datos exigía intervención antes de cualquier modelado: se corrigieron o
  documentaron **más de 600 registros problemáticos** distribuidos en 6 variables distintas.
- **El diccionario de datos no fue documentación decorativa**: reveló tres errores metodológicos
  reales (correlación de Pearson sobre variable nominal, rango de validación sin fuente, y ceros
  falsos tratados como scores) y produjo dos hallazgos nuevos.
- No existe un predictor individual suficiente. El valor está en la combinación de variables y en
  los atributos derivados, lo que justifica un modelo multivariado en la siguiente fase.
- El desbalance de clases (20:1) condiciona el diseño completo del modelado y la elección de
  métricas.

### Recomendaciones de negocio

1. Incorporar tres criterios de alerta temprana en la originación: **mora previa**, **ausencia de
   historial crediticio** y **tipo de crédito 6**.
2. Auditar el origen de los datos: los errores hallados apuntan a fallas de captura o exportación
   que conviene corregir en la fuente, no solo en el análisis.
3. Investigar con el área de producto qué representa el tipo de crédito 6.
4. Confirmar la temporalidad de los saldos en mora antes de usarlos en un modelo.

## Limitaciones

| Limitación | Implicación |
|---|---|
| **Sin identificador de cliente** | La unidad de observación es el crédito, no la persona. El 67.9% de las filas comparte perfil demográfico con otra. Impide split agrupado por cliente → riesgo de fuga entre train y test |
| **Posible fuga de información** | `saldo_mora` y `saldo_mora_codeudor` podrían registrarse después del desembolso |
| **Muestras pequeñas** | Tipo de crédito 6 (21 registros) y mora previa (55 registros): señales indicativas, no concluyentes |
| **27% de nulos sin causa estructural** | En `promedio_ingresos_datacredito`, sin patrón que justifique imputación |
| **Códigos de `tipo_credito` sin confirmar** | No corresponden a ninguna clasificación oficial de la SFC |
| **Sin diccionario de datos original** | La interpretación de varias variables se basa en investigación e inferencia documentada |


proximos pasos: requisitos ya definidos a partir del análisis

- **Problema:** clasificación binaria supervisada.
- **Desbalance 20:1** →  usar `class_weight='balanced'`, submuestreo o SMOTE,
  aplicado **solo sobre el conjunto de entrenamiento**.
- **Métricas:** precisión, recall, F1 y AUC-PR sobre la clase minoritaria. **La exactitud
  (accuracy) queda descartada ya que un modelo trivial alcanzaría 95.25% sin aprender nada.
- **Advertencia de split:** sin identificador de cliente no es posible un `GroupShuffleSplit`;
  debe declararse como limitación al reportar métricas.
- **Interpretabilidad:** requisito de negocio en riesgo crediticio (hay que poder explicar por
  qué se niega un crédito) 

## Estructura del repositorio

```
MLOPS_CURSE/
├── src/
│   ├── transformacion_eda.ipynb      # Fase 1: diccionario, limpieza, EDA completo (COMPLETADO)
│   ├── config.json                   # Configuración del proyecto
│   ├── ft_engineering.py             # Fase 2: Feature Engineering (pendiente)
│   ├── model_training_evaluation.py  # Fase 3: Entrenamiento y evaluación (pendiente)
│   ├── model_deploy.py               # Fase 4: Despliegue (pendiente)
│   └── model_monitoring.py           # Fase 5: Monitoreo (pendiente)
├── Base_de_datos.csv                 # Datos crudos originales (no modificar)
├── Base_de_datos_limpia.csv          # Salida de la Fase 1 (generado por el notebook)
├── PRESENTACION.pptx                 # Presentación de insights
├── requirements.txt                  # Dependencias
├── set_up.bat                        # Script de instalación de dependencias
├── .gitignore
└── README.md
```

### Ramas

| Rama | Propósito |
|---|---|
| `developer` | Trabajo activo y experimentación |
| `master` | Versión estable (merge al cerrar cada fase validada) |
| `certification` | Entregable final |

## Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.11 | Lenguaje base |
| pandas, numpy | Manipulación y análisis de datos |
| matplotlib, seaborn | Visualización |
| scipy | Estadística (skewness, kurtosis, chi-cuadrado) |
| scikit-learn | PCA y estandarización |
| Jupyter | Notebooks de análisis |
| Git / GitHub | Control de versiones (3 ramas) |

## Instrucciones de ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/papiarcacamilo/MLOPS_CURSE.git
cd MLOPS_CURSE

# 2. Instalar dependencias
#    Windows:
set_up.bat
#    Linux / macOS:
pip install -r requirements.txt

# 3. Ejecutar el análisis
jupyter notebook src/transformacion_eda.ipynb
#    (o abrir la carpeta en VS Code y usar "Run All")
```

**Reproducibilidad:** el notebook se ejecuta de principio a fin sin errores. Todas las rutas son
relativas (`../Base_de_datos.csv` desde `src/`). El archivo crudo nunca se modifica; la salida
`Base_de_datos_limpia.csv` se regenera en cada ejecución completa.

## Referencias

1. **DataCrédito Experian** — Términos y Condiciones Midatacrédito. Rango oficial del score
   crediticio (150–950). https://www.datacredito.com.co
2. **Ley 1266 de 2008** — Disposiciones generales del hábeas data y manejo de información
   financiera, crediticia y comercial. Congreso de la República de Colombia.
   http://www.secretariasenado.gov.co/senado/basedoc/ley_1266_2008.html
3. **Ley 1581 de 2012** — Disposiciones generales para la protección de datos personales.
   https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981
4. **Superintendencia Financiera de Colombia** — Modalidades de crédito y clasificación de
   cartera. https://www.superfinanciera.gov.co
5. **Superintendencia de Industria y Comercio** — Manejo de información personal, Habeas Data.
   https://www.sic.gov.co/manejo-de-informacion-personal
6. **Banco de la República** — Reporte de la situación del crédito en Colombia.
   https://www.banrep.gov.co
7. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. (Método IQR para detección
   de atípicos)

---

**Autor:** Camilo Andrés Armenta — [@papiarcacamilo](https://github.com/papiarcacamilo)

**Docente:** Juan Sebastián Parra Sánchez — Ciencia de Datos en Producción
