# MLOPS_CURSE — Análisis de Riesgo Crediticio

Proyecto transversal de **Ciencia de Datos en Producción**. Construye un pipeline MLOps completo
sobre una base de datos real de créditos de una empresa financiera colombiana, desde la
comprensión y limpieza de los datos hasta el despliegue y monitoreo de un modelo predictivo.

> **Estado actual:** las cinco fases cerradas. El modelo está elegido, evaluado sobre test una sola
> vez, auditado en calibración y *fairness*, traducido a scorecard, servido en un endpoint por lote
> dentro de una imagen Docker, y monitoreado por cosecha sobre el registro que ese endpoint produce.
> Este readme describe únicamente lo que el código implementa hoy.

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
11. [Ingeniería de características](#ingeniería-de-características)
12. [Modelamiento](#modelamiento)
13. [Despliegue](#despliegue)
14. [Monitoreo](#monitoreo)
15. [Resultados](#resultados)
16. [Conclusiones](#conclusiones)
17. [Limitaciones](#limitaciones)
18. [Estructura del repositorio](#estructura-del-repositorio)
19. [Tecnologías utilizadas](#tecnologías-utilizadas)
20. [Instrucciones de ejecución](#instrucciones-de-ejecución)
21. [Referencias](#referencias)

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
| Variables tras limpieza | 35 (incluye columnas de trazabilidad) |
| Separador | `;` |
| Codificación | UTF-8 con BOM |
| Rango temporal | 2024-11-26 a 2026-04-26 |
| Variable objetivo | `Pago_atiempo` (1 = al día, 0 = mora) |
| Distribución del objetivo | 95,25% al día / 4,75% mora (ratio 20:1) |

**Fuente de los datos:** base de datos entregada por el docente del curso, correspondiente a
operaciones de crédito de una entidad financiera. No se proporcionó documentación adjunta.

**Unidad de observación:** el crédito, no el cliente. El dataset **no contiene identificador de
cliente** (ver [Limitaciones](#limitaciones)).

## Diccionario de datos

El diccionario está implementado como estructura de código en
`mlops_pipeline/comprension_eda.ipynb`
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
- **`saldo_mora` y `saldo_mora_codeudor`**: **fuga de información CONFIRMADA por el negocio** — se
  registran *después* del desembolso, por lo que son consecuencia y no causa del impago. Se excluyen
  del modelado desde la Fase 2, junto con `tiene_mora_previa` y las banderas derivadas de ellas. Por
  precaución se excluyen también `saldo_total` y `saldo_principal`, del mismo corte del buró (IV de
  0,018 a 0,021: sin coste apreciable). Las cinco variables de mayor IV están libres de fuga.

### Columnas de trazabilidad generadas

El proceso de limpieza añade columnas booleanas que documentan qué registros fueron modificados,
permitiendo auditar cada intervención:

`edad_cliente_corregida`, `salario_cliente_corregido`, `tendencia_ingresos_reconstruida`,
`sin_historial_crediticio`, `total_otros_prestamos_sospechoso`, `saldo_principal_era_nulo`,
`saldo_mora_era_nulo`, `saldo_total_era_nulo`, `saldo_mora_codeudor_era_nulo`,
`promedio_ingresos_datacredito_era_nulo`.

> **Estas columnas dejaron de ser solo auditoría.** Al cruzarlas con la variable objetivo
> (sección 3.2.3 del notebook), **tres resultan predictores estadísticamente significativos**:
> `promedio_ingresos_datacredito_era_nulo`, `saldo_mora_codeudor_era_nulo` y
> `saldo_principal_era_nulo`. Deben conservarse como variables del modelo en la Fase 2.

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
| `puntaje` con 87,4% de valores idénticos | 9.407 | Columna eliminada | Sin capacidad discriminante; se comporta como constante |
| `fecha_prestamo` como texto | 10.763 | Conversión a `datetime` | Permite análisis temporal |
| Edades imposibles (121–123 años) | 150 | Imputación con mediana (42) | Salto claro en la distribución: sin registros entre 70 y 120 |
| Salarios extremos y en cero | 250 | Imputación con mediana ($3M) | IQR sobre escala logarítmica (técnica para variables monetarias asimétricas) |
| `tendencia_ingresos` con valores numéricos | 58 | Reconstrucción por signo | Los 58 tenían `promedio_ingresos_datacredito` poblado: fallo de categorización en origen |
| **Score fuera del rango [150, 950]** | **153** | **Marcado como `sin_historial_crediticio`, score → NaN** | **No son scores bajos: son ausencia de score (fuente: DataCrédito Experian)** |
| Nulos en saldos | 405 / 590 | Imputación con 0 | Causa estructural confirmada: `cant_creditosvigentes = 0` |
| Nulos en `puntaje_datacredito` | 6 | Imputación con mediana | Impacto insignificante (0,06%) |
| Nulos en `promedio_ingresos_datacredito` | 2.930 | **No imputados** (NaN + bandera) | 27% sin causa estructural: imputar fabricaría el dato |
| Nulos en `tendencia_ingresos` | 2.932 | Categoría explícita `Sin_dato` | Preserva la información de ausencia |

**Salida:** `Base_de_datos_limpia.csv` (10.763 × 35: 22 originales + 9 de trazabilidad + 4 derivadas del análisis temporal). El archivo crudo `Base_de_datos.csv`
permanece intacto para garantizar reproducibilidad desde cero.

## Análisis exploratorio

### Análisis univariado

- **Cuantitativas:** `describe()` completo, medidas de **tendencia central** (media, mediana,
  **moda** con su frecuencia, mínimo y máximo) y de **dispersión** (rango, IQR, cuartiles Q1/Q2/Q3,
  varianza, desviación estándar, **skewness**, **kurtosis**), histogramas de 11 variables clave y
  boxplots comparativos de edad y salario **antes y después de la limpieza**.
- **Cualitativas:** tablas de frecuencia absoluta y relativa, countplots de las 4 variables
  nominales.
- **Tablas pivote:** resumen de variables numéricas agregadas por `tipo_credito` y por
  `tipo_laboral`.

**Hallazgos de forma de distribución:**

| Variable | Skewness | Kurtosis | Distribución |
|---|---|---|---|
| `edad_cliente` | 0,27 | −0,83 | Aproximadamente simétrica / cuasi-gaussiana |
| `capital_prestado` | 3,72 | 35,3 | Fuertemente sesgada a la derecha (log-normal) |
| `salario_cliente` | 2,20 | 6,34 | Sesgada a la derecha (log-normal) |
| `puntaje_datacredito` | −0,71 | 5,25 | Unimodal, moderadamente sesgada a la izquierda |
| `saldo_mora_codeudor` | 97,7 | 9.813 | Extremadamente sesgada (mayoría en 0) |

> **Nota sobre `puntaje_datacredito`.** Antes de la depuración esta variable presentaba una forma
> bimodal, con un grupo artificial de 145 registros en el valor 0. Ese grupo no eran scores bajos
> sino ausencia de score, y fue convertido a NaN en la sección 2.4 del notebook. Tras la corrección
> la distribución es unimodal (mínimo 287, máximo 947). **La bimodalidad era un artefacto del dato
> sucio, no una característica de la variable.**

### Análisis bivariado

- **Cuantitativa vs. objetivo:** comparación de medianas por grupo y boxplots comparativos de 6
  variables clave.
- **Cualitativa vs. objetivo:** tasas de mora por categoría (barplots) y **prueba chi-cuadrado
  de independencia con V de Cramér** — técnica correcta para variables nominales, en lugar de
  correlación de Pearson.

**Resultados de la prueba chi-cuadrado:**

| Variable | χ² | gl | p-valor | V de Cramér | ¿Asociación significativa? |
|---|---|---|---|---|---|
| `tipo_credito` | 68,86 | 5 | 1,77e−13 | 0,0800 | Sí |
| `tendencia_ingresos` | 21,48 | 3 | 8,38e−05 | 0,0447 | Sí |
| `tipo_laboral` | 8,00 | 1 | 0,0047 | 0,0273 | Sí |

#### Análisis complementarios del bivariado

Tres subsecciones adicionales cubren aspectos que la comparación de medianas y la correlación de
Pearson no detectan:

**3.2.1 — Tasa de mora por tramos.** Con un target binario al 4,75%, la correlación lineal
subestima sistemáticamente el poder discriminante de una variable. Segmentar en tramos lo revela:

| Variable | Rango de tasa de mora por tramo | Correlación de Pearson |
|---|---|---|
| `puntaje_datacredito` | **64,00%** (<600 pts) → **2,91%** (>850 pts) | 0,1212 |
| `plazo_meses` | 3,91% (6–12 m) → **12,66%** (24–36 m) | 0,0631 |

En ambos casos la conclusión inicial basada en medianas o correlación era incorrecta. `plazo_meses`
llegó a describirse como "sin diferencia" porque las medianas de ambos grupos empatan en 10 meses:
la señal está en la cola, no en el centro.

**3.2.2 — Comportamiento temporal de la variable objetivo.** La tasa de mora varía entre 1,72% y
9,09% según el mes de desembolso, y el **19,9% de los créditos tiene un plazo pactado que vence
después de la última fecha del dataset** (tasa de mora 6,40% vs 4,34%, p = 0,0001). Para esos
registros, `Pago_atiempo` refleja el estado al corte y no el desenlace final. Consecuencias: debe
evaluarse un split temporal en la Fase 3 y esta serie es la línea base de monitoreo de la Fase 5.

**3.2.3 — Las banderas de trazabilidad como predictores.** Las columnas de auditoría creadas
durante la limpieza se cruzaron con la variable objetivo. Tres resultan significativas:

| Bandera | n | Tasa de mora | vs. resto | p (Fisher) |
|---|---|---|---|---|
| `promedio_ingresos_datacredito_era_nulo` | 2.930 | 5,73% | 4,38% | **0,0037** |
| `saldo_mora_codeudor_era_nulo` | 590 | 6,78% | 4,63% | **0,0216** |
| `saldo_principal_era_nulo` | 405 | 7,16% | 4,65% | **0,0308** |

Las tres describen el mismo perfil: clientes con menor huella en el sistema financiero formal.
**Deben conservarse como variables del modelo en la Fase 2, no descartarse como metadatos.**

#### Rigor estadístico aplicado

Todo hallazgo apoyado en un grupo pequeño se acompaña de su **prueba exacta de Fisher** y su
**intervalo de confianza de Wilson al 95%** (función `contraste_tasas()`, reutilizada en todo el
notebook). Los que no alcanzan significancia se presentan explícitamente como hipótesis. El
criterio es preferir menos hallazgos y que sean sólidos.

**Cada gráfica del notebook incluye una celda de observaciones escritas debajo**, con la lectura
concreta de lo que muestra, las cifras que la sustentan y su implicación para las fases
siguientes. Son 10 bloques de interpretación: boxplots antes/después, histogramas, countplots,
gráficos bivariados, tasas por tramo, evolución temporal, banderas de trazabilidad, matriz de
correlación, pairplot y gráficos de dispersión.

### Análisis multivariado

- **Matriz de correlación** sobre variables cuantitativas (excluye nominales por definición del
  diccionario) + detección automática de multicolinealidad.
- **Pairplot** con `hue` en la variable objetivo (muestra de 1.500 registros).
- **Gráficos de dispersión** entre pares de variables numéricas, coloreados por la variable
  objetivo.
- **Tablas de contingencia** cruzadas entre variables categóricas.

**Correlación con la variable objetivo (top 5):**

| Variable | \|r\| |
|---|---|
| `puntaje_datacredito` | **0,1212** |
| `huella_consulta` | 0,0737 |
| `saldo_mora` | 0,0726 |
| `plazo_meses` | 0,0631 |
| `edad_cliente` | 0,0524 |

**Multicolinealidad detectada (|r| > 0,7):**

| Par | r | Causa |
|---|---|---|
| `cant_creditosvigentes` ↔ `creditos_sectorFinanciero` | +0,791 | El sector financiero concentra la mayoría de los créditos del cliente |
| `capital_prestado` ↔ `cuota_pactada` | +0,764 | Relación mecánica: a mayor monto, mayor cuota |
| `saldo_total` ↔ `saldo_principal` | +0,737 | El principal es un componente del total |

**Separabilidad:** el pairplot sobre variables de perfil y monto no muestra ninguna región donde
se concentre la mora; las nubes se superponen ampliamente. **La excepción está en los gráficos de
dispersión:** la banda de `puntaje_datacredito` por debajo de 600 puntos concentra un 64% de mora
(25 créditos, 16 en mora). Es la única región del EDA con separación visual clara. Fuera de ella,
el riesgo depende de la combinación de varios factores, lo que justifica un modelo multivariado en
la Fase 3.

> **Advertencia sobre la lectura de la tabla de correlaciones.** Los valores por debajo de 0,13
> significan que **ninguna relación es lineal**, no que no exista relación. Las secciones 3.2.1 y
> 3.2.2 demuestran que las mismas variables sí discriminan riesgo cuando se analizan por tramos.

## Reglas de validación

Implementadas como diccionario `REGLAS_VALIDACION` y función `validar_dataframe()`, reutilizables
en fases posteriores del pipeline y para validar datos nuevos en producción. Cada regla incluye
su campo `fuente`.

**Cobertura: las 22 variables del dataset.** Una versión anterior dejaba fuera 6 variables
(`fecha_prestamo`, `total_otros_prestamos`, los tres conteos por sector y
`promedio_ingresos_datacredito`), que pasaban la validación sin ser revisadas. La ausencia de regla
temporal era especialmente relevante: en producción, una fecha de desembolso futura es el error de
captura más común y no habría sido detectado.

**Corrección relevante:** la regla de `puntaje_datacredito` era `[0, 1000]` — un rango sin
fuente. Corregida a `[150, 950]` (rango oficial DataCrédito Experian). La regla anterior detectaba
**1** registro inválido; la corregida detecta **153**.

**Diagnóstico añadido — `total_otros_prestamos` (sección 4.1 del notebook).** Es la única variable
monetaria que no pasó por la limpieza de la sección 2. Alcanza un máximo de $6.787 millones, con 13
registros por encima de mil millones y 30 donde el endeudamiento supera 100 veces el salario
mensual. **No se imputan:** a diferencia del salario, aquí no existe una regla de negocio que
permita distinguir un endeudamiento corporativo legítimo de un error de digitación. Se marcan con
la bandera `total_otros_prestamos_sospechoso` y se define una regla con techo para que en
producción queden señalados en lugar de pasar silenciosamente.

## Ingeniería de características

> **Estado: identificada y evaluada, NO implementada.** El requerimiento del curso pide
> *identificar* atributos derivados durante el EDA (marcado como "MUY IMPORTANTE Y DE GRAN
> VALOR"), no construirlos. Se calculan en un DataFrame independiente (`derivadas = df.copy()`)
> solo para medir su poder predictivo; **ninguno se incorpora a `Base_de_datos_limpia.csv`**.
> Su implementación definitiva corresponde a la Fase 2
> (`mlops_pipeline/ft_engineering.py`, hoy vacío).

Se propusieron y evaluaron 9 atributos derivados dentro del notebook, midiendo su relación con la
variable objetivo mediante **dos medidas complementarias**: correlación de Pearson (relación
lineal) e **Information Value** (capacidad de discriminación por tramos, sin asumir linealidad —
la medida estándar en construcción de *scorecards* crediticios).

| Atributo derivado | IV | \|r\| | Evaluación |
|---|---|---|---|
| `consultas_por_credito` | **0,1637** | 0,077 | **Poder medio — el derivado más valioso.** Supera a `huella_consulta` original (IV 0,1455). Tasa de mora de 2,78% a 7,82% por quintil |
| `discrepancia_ingresos` | 0,0930 | 0,038 | Débil-alto, cerca del umbral medio. Calculable solo en el 73% de los registros |
| `tiene_mora_previa` | 0,0884 | **0,107** | 36,36% vs 4,59% de mora (OR ≈ 12, p < 0,0001). Ver matiz abajo |
| `ratio_deuda_ingreso` | 0,0726 | 0,007 | Débil, pero **no nulo**: evaluar por tramos |
| `mes_prestamo` | 0,0430 | 0,009 | Débil; conservar por el hallazgo temporal de 3.2.2 |
| `ratio_capital_ingreso` | 0,0353 | 0,0002 | Débil |
| `ratio_cuota_ingreso` | 0,0350 | 0,003 | Débil |

Como referencia, `puntaje_datacredito` (variable original) alcanza IV = 0,2136 y sigue siendo el
predictor individual más fuerte del dataset.

> **Matiz sobre `tiene_mora_previa` que la correlación oculta.** Es el derivado con mayor
> correlación (0,107) y una diferencia de tasas enorme, pero su Information Value es **débil**
> porque **solo toca 55 registros, el 0,5% de la cartera**. Su aporte a la capacidad global de
> discriminación de un modelo es marginal aunque el riesgo individual sea altísimo.
> **Es una regla de alerta de originación, no un predictor de volumen.** Debe incorporarse por su
> valor de negocio e interpretabilidad, sin esperar que mejore las métricas globales.

**Sobre los ratios de capacidad de pago.** Una versión anterior de este análisis atribuía su
correlación casi nula a los 250 salarios imputados con la mediana durante la limpieza. **Esa
hipótesis se contrastó directamente y resultó falsa:** al recalcular las correlaciones excluyendo
esos registros, los valores no se mueven (de 0,0028 a 0,0027 en `ratio_cuota_ingreso`). Con 250
registros sobre 10.763 —el 2,3%— era matemáticamente improbable que fuera el factor determinante.

La explicación real es metodológica: son cocientes fuertemente asimétricos (máximos de 126, 2.439
y 840) evaluados contra un target binario al 4,75%, condiciones en las que Pearson no detecta nada
aunque exista estructura. El IV lo confirma: `ratio_deuda_ingreso` pasa de |r| = 0,007 a IV =
0,073. Sigue siendo débil, pero **no es cero**.
*Recomendación corregida para la Fase 2:* no recalcularlos excluyendo salarios imputados —está
demostrado que no cambia nada— sino **evaluarlos por tramos** y revisar el efecto de los extremos
de `total_otros_prestamos` documentados en la sección 4.1.

**Transformaciones planificadas** (documentadas en la sección 6 del notebook): One-Hot Encoding
para categóricas, transformación logarítmica para variables monetarias asimétricas, `log1p` para
variables con ceros, escalado robusto, extracción de componentes temporales y —añadido a partir de
los hallazgos de 3.2.1— **binning por tramos de `puntaje_datacredito` y `plazo_meses`**, cuya
relación con la mora no es lineal.

## Modelamiento

> **Estado: etapas 1 a 5 completadas.** Piso heurístico fijado
> (`hueristic_model.py`) y modelo seleccionado (`model_training.py`).
> Pendientes la evaluación final, el despliegue y el monitoreo.

**Problema:** clasificación binaria supervisada en originación, con desbalance de 20:1.

**Métrica principal AUC-PR.** La exactitud queda descartada: un modelo trivial alcanzaría
95,25% sin aprender nada. Se acompaña de KS, Gini, Brier y recall en el punto de operación.

**Sin identificador de cliente** no es posible un split agrupado. Debe declararse como
limitación junto a cualquier métrica que se publique.

### El piso: qué se consigue sin modelo

Regla de negocio sobre `puntaje_datacredito` con corte en 750, derivada del principio
*rechazar donde el riesgo observado supera el promedio de la cartera*. No se optimiza ninguna
métrica: hacerlo la convertiría en un modelo entrenado y dejaría de ser un piso honesto.

| | Estratificado | Temporal |
|---|---|---|
| AUC-PR del score como ordenador | 0,0753 | 0,0862 |
| Rechaza | 20,9% | 20,8% |
| Mora capturada | 37,2% | 38,2% |

### Selección del modelo

Nueve configuraciones: tres familias por tres tratamientos del desbalance, con la ingeniería
de características **dentro** de la validación cruzada.

| Modelo | Desbalance | AUC-PR | Desv. folds | Brier |
|---|---|---|---|---|
| **logística** | **ninguno** | **0,1385** | 0,0159 | **0,0437** |
| logística | smote | 0,1363 | 0,0147 | 0,2153 |
| logística | class_weight | 0,1323 | 0,0122 | 0,2158 |
| bosque | ninguno | 0,1287 | 0,0241 | 0,0442 |
| boosting | ninguno | 0,1109 | 0,0283 | 0,0456 |

**Dos resultados que corrigen suposiciones previas de este documento.**

*Los árboles no superan a la logística.* Las tres configuraciones de boosting quedan por debajo
de todas las de la logística. El WoE ya codifica la no linealidad de cada variable, que es para
lo que se eligió en la Fase 2, así que los árboles no encuentran estructura nueva y ajustan
ruido. Se ve en su desviación entre folds, más del doble.

*El tratamiento del desbalance no era obligatorio.* El rango de AUC-PR entre los tres
tratamientos es 0,0062, menor que la desviación entre folds (0,0159): la diferencia es ruido.
Pero `class_weight` y SMOTE empeoran el Brier cinco veces, porque desplazan las probabilidades
por diseño. Sin ganancia en discriminación, solo añaden un paso de recalibración.

SMOTE se midió en lugar de descartarlo por argumento, aunque el argumento era sólido: interpola
entre valores WoE y produce clientes sintéticos que no corresponden a ningún tramo real.

**Modelo seleccionado: regresión logística sobre WoE, sin tratamiento del desbalance.** Las dos
particiones lo eligen de forma independiente. Serializado en
`data/models/modelo_seleccionado.joblib`, con la ingeniería incluida: predice desde registros
crudos.

**Revisión de signos.** Sobre entrenamiento, los 9 coeficientes WoE salen positivos y su orden
reproduce el ranking de Information Value de la Fase 1. Es la comprobación obligatoria antes de
firmar un scorecard, pero por sí sola **no prueba que la dirección generalice**: el WoE se ajusta
sobre entrenamiento y los coeficientes se aprenden sobre entrenamiento, así que su signo solo
confirma que el modelo aprendió la dirección que uno mismo codificó.

Comprobado fuera de los datos de ajuste:

| | Resultado |
|---|---|
| Folds con los 9 positivos | **5 de 5** |
| Positivos en test, con la receta congelada | **7 de 9** |

Los dos que se invierten tienen lecturas distintas. `woe_tipo_credito_grp` cae a **−0,0066**,
indistinguible de cero: las categorías que cargan la señal tienen 3 y 1 registros en test.
`woe_discrepancia_ingresos` sí se invierte de verdad, de +0,15 a **−0,40**, y el detalle univariado
confirma que su gradiente monótono en entrenamiento (2,79% → 6,18%) desaparece en test.

El contexto que hay que declarar: el test tiene **6 eventos por variable** frente a los 24 de
entrenamiento, muy por debajo del mínimo de 10 que se considera necesario para que los coeficientes
de una logística sean estables. Ajustados sobre test son ruidosos por construcción.

### Contra el piso, en su mismo punto de operación

Rechazando el mismo 20,9% de solicitudes:

| | Regla | Modelo |
|---|---|---|
| Mora capturada | 37,2% | **44,5%** |
| Precisión | 8,44% | **10,12%** |

**El conjunto de prueba no se ha utilizado.** Toda la selección se hizo con validación cruzada
sobre entrenamiento. El test se abre una sola vez, en `model_evaluation.py`.

### Evaluación final

El conjunto de prueba se abrió **una sola vez**, con el modelo ya elegido y el umbral ya fijado.
Calibración, *fairness*, umbral y scorecard se decidieron antes, sobre predicciones fuera de fold.

| | Modelo | Piso heurístico en el **mismo** test | Ventaja |
|---|---|---|---|
| Estratificado | **0,1479** (lift 3,12×) | 0,0829 (1,75×) | **1,78×** |
| Temporal | **0,0647** (lift 2,02×) | 0,0445 (1,39×) | **1,45×** |

El modelo supera al piso en las dos particiones, así que **pasa la prueba de estrés temporal**.
Su ventaja cae de 1,78× a 1,45×: degrada bajo desplazamiento temporal, pero sigue aportando.

Comparar el AUC-PR entre particiones directamente sería un error, porque depende de la tasa base
y esta difiere (4,74% frente a 3,20%). Por eso se evalúa el piso sobre el mismo conjunto y se
reporta el *lift*, que sí es comparable.

| Métrica | Estratificado | Temporal |
|---|---|---|
| KS | 0,2742 | 0,2026 |
| Gini | 0,3697 | 0,2415 |
| Brier | 0,0434 | 0,0333 |

**KS queda por debajo de 0,30**, el umbral que se considera utilizable en un scorecard. Es una
limitación que hay que declarar: el modelo aporta sobre la regla, pero no alcanza el estándar del
sector para operar sin supervisión.

### Calibración: no hace falta corregirla

Sobre predicciones fuera de fold, la probabilidad media predicha es **0,04742** frente a una mora
observada de **0,04750**. El sesgo es de ocho diezmilésimas y el error de calibración esperado
(ECE) queda en 0,00812.

Es el pago directo de haber elegido *ninguno* en el tratamiento del desbalance: `class_weight` y
SMOTE dejaban un Brier cinco veces peor y habrían obligado a un paso de Platt o isotónica.
Recalibrar un modelo ya calibrado solo añadiría varianza, así que **no se aplica**.

Importa porque la probabilidad de incumplimiento entra al cálculo de provisiones: mal calibrada
significa provisionar mal, que es un problema contable antes que estadístico.

### Fairness: dos hallazgos que hay que poder defender

Se miden tasa de rechazo y tasas de error **por grupo**, no solo el AUC global.

**`tipo_laboral`**, retirado del modelo en la selección de variables:

| | Riesgo real | Tasa de rechazo |
|---|---|---|
| Empleado | 4,31% | 18,61% |
| Independiente | 5,50% | 24,82% |

La brecha de rechazo (6,21 puntos) es **cinco veces** la brecha de riesgo real (1,19 puntos).
**Retirar la variable no eliminó el sesgo**, exactamente como se advirtió al decidirlo: otras
variables actúan de *proxy* de la informalidad laboral. Queda medido, no supuesto.

**`edad_cliente`**, que sí está en el modelo:

| | Riesgo real | Tasa de rechazo |
|---|---|---|
| 18-30 | 6,76% | **41,94%** |
| 31-45 | 4,80% | 22,51% |
| 46-60 | 3,64% | 10,20% |
| 60+ | 4,32% | **8,39%** |

Los jóvenes son rechazados **cinco veces más** que los mayores cuando su riesgo real es 1,6 veces
mayor. La brecha de igualdad de opciones (TPR) llega a 43 puntos.

Se documenta sin corregirlo: **es una decisión de negocio, no técnica**. Las opciones son umbrales
por grupo, retirar la edad a costa de rendimiento, o asumirlo con justificación explícita. Ninguna
es una elección que corresponda al modelador tomar en solitario.

### Punto de operación

El umbral se fija por **volumen de rechazo**, no por probabilidad: se elige el que rechaza el mismo
20,9% que la regla heurística, para que la comparación sea a igual coste comercial. El 0,5 por
defecto no sirve aquí, porque sin reponderar clases el modelo casi nunca lo supera.

| | Rechaza | Mora capturada | Precisión |
|---|---|---|---|
| Fuera de fold, entrenamiento | 20,9% | 43,3% | 9,83% |
| Test estratificado | 21,6% | 40,2% | 8,84% |
| Test temporal | **33,3%** | 42,0% | 4,04% |

**Un umbral fijo no se sostiene cuando la población se desplaza.** En el test temporal, el mismo
umbral rechaza el 33,3% en lugar del 21%. Es la señal que tendrá que vigilar el monitoreo.

### Scorecard

Tabla de 45 filas con puntaje base 601 y puntajes observados entre 464 y 767, en la escala
convencional del sector (PDO 20, base 600 a odds 20:1). Más puntos significan menos riesgo.

Verificado que **la tabla reproduce la predicción del modelo con error menor a 1e-9**: si no lo
hiciera, el analista estaría viendo una tabla que no representa lo que decide el sistema.

Solo las 9 variables WoE se tabulan por tramo. Las 4 monetarias escaladas son continuas y entran
como ajuste sobre el puntaje, no como fila de la tabla.


## Despliegue

El enunciado pide tres piezas, y cada una vive en su archivo.

| Pieza | Archivo | Qué hace |
|---|---|---|
| La app | `mlops_pipeline/app.py` | API con los endpoints, capa HTTP únicamente |
| La imagen | `Dockerfile` | Librerías, código y artefactos, sin datos de entrenamiento |
| El motor | `mlops_pipeline/model_deploy.py` | Cadena de inferencia y decisión |

### Cadena de inferencia

`sanear` → `validar` → `transformar` → `puntuar` → `decidir` → `registrar`

El requisito que no puede fallar es que un cliente nuevo atraviese exactamente las mismas
transformaciones que el conjunto de entrenamiento. Si el endpoint recalcula cortes, reajusta WoE o
imputa de otra forma, el modelo recibe una entrada que no se parece a lo que aprendió y falla en
silencio: sigue devolviendo probabilidades, solo que equivocadas.

Por eso aquí nada se ajusta. `cadena_inferencia()` reutiliza los pasos ya ajustados del `.joblib` y
les antepone las derivadas, que son fila a fila. Y las constantes de saneamiento se congelan en el
artefacto en lugar de recalcularse sobre el lote entrante: una mediana estimada sobre 50 solicitudes
no es la mediana con la que el modelo aprendió.

### Por qué sanear va antes de validar

El orden previsto era el inverso, y al implementarlo no funciona. El contrato exige
`puntaje_datacredito` entre 150 y 950, pero la Fase 1 documentó que un valor fuera de ese rango no
es un error: es ausencia de historial, y su tratamiento es pasar a nulo con bandera. Validar primero
rechazaría como inválido justo lo que el EDA decidió conservar.

La distinción es entre dos clases de anomalía.

| Clase | Ejemplos | Qué hace el endpoint |
|---|---|---|
| Con tratamiento definido | Edad sobre 90, salario cero o extremo, tendencia corrupta, score fuera de rango | Repite el tratamiento de la Fase 1 y puntúa |
| Sin tratamiento definido | Columna ausente, edad de 12 años, plazo 0, `tipo_credito` 99 | Rechazo técnico, sin puntuar |

Una excepción, y hay que justificarla: el contrato pone techo de 1.000 millones a
`total_otros_prestamos` para **señalar** los 13 registros no verificables, no para excluirlos. La
Fase 1 decidió marcarlos sin imputar, y esos 13 registros entraron al entrenamiento con su bandera
puesta. Aplicar el techo como rechazo técnico negaría en producción lo que en entrenamiento se
puntuó, y dejaría `total_otros_prestamos_sospechoso` sin poder activarse nunca por esa vía. El
mínimo sí se conserva: una deuda negativa no tiene tratamiento definido.

### Dos verificaciones que se ejecutan en cada corrida

| Verificación | Qué comprueba | Resultado |
|---|---|---|
| `verificar_saneamiento()` | Sanear el crudo reproduce `Base_de_datos_limpia.csv` | 10.763 registros, 10 columnas, 0 diferencias |
| Equivalencia de cadenas | Partir del crudo da la misma probabilidad que partir del limpio | Error máximo 0.00e+00 |

La segunda es la prueba definitiva contra el *train-serving skew*: la ruta del endpoint y la ruta
que se evaluó son numéricamente idénticas sobre los 10.763 registros.

### Decisión y política

El umbral **0,0661** no se elige en despliegue: viene de `model_evaluation.py`, fijado sobre
predicciones fuera de fold para rechazar el mismo 20,9% que rechazaba la regla heurística. Elegirlo
aquí, con los datos que llegan, sería ajustar el punto de operación a la población que se quiere
medir.

`RECHAZAR_SIN_SCORE` es política de negocio publicada en el contrato, no una salida del modelo.
Medido sobre la población completa:

| Origen del rechazo | Porcentaje |
|---|---|
| Umbral del modelo | 21,22% |
| Política, casos que el umbral no cubría | 0,40% |
| Rechazo total | 21,62% |

El modelo ya penaliza la ausencia de score (WoE +0,417), de modo que la política solo agrega ese
0,40%. Se aplica después de puntuar y la probabilidad se reporta igual, con el motivo declarando la
política: sin eso, la respuesta mostraría una probabilidad baja junto a un rechazo y nada que
explicara la contradicción.

### Bandas de riesgo

Los cortes son los quintiles del puntaje en train, no números elegidos a mano, y cada banda se
publica con la tasa de mora que le corresponde.

| Banda | Puntaje | Registros en train | Mora observada |
|---|---|---|---|
| A | 627,1 o más | 1.722 | 1,51% |
| B | 614,2 a 627,1 | 1.722 | 2,56% |
| C | 603,0 a 614,2 | 1.722 | 3,37% |
| D | 588,9 a 603,0 | 1.722 | 5,23% |
| E | Menos de 588,9 | 1.722 | 11,09% |

De A a E la mora se multiplica por 7,3. Es la lectura que un comité de crédito puede usar sin
entender el modelo.

### Endpoints

| Método | Ruta | Devuelve |
|---|---|---|
| `GET` | `/salud` | Sonda de vida, usada por el `HEALTHCHECK` del contenedor |
| `GET` | `/modelo` | Versión, umbral, desempeño en test, esquema de entrada y bandas |
| `GET` | `/scorecard` | Tabla de puntos por tramo |
| `POST` | `/predecir` | Lote en JSON |
| `POST` | `/predecir/archivo` | Lote en CSV, respuesta en CSV |

La respuesta trae `decision`, `probabilidad_mora`, `puntaje`, `banda` y `motivo`. Ningún campo
queda vacío: una solicitud rechazada por contrato trae la regla que incumplió, y una puntuada trae
las tres variables que más mueven su puntaje.

```
APROBAR          p=0.039792  pts=605.4  C          plazo_meses -25; discrepancia_ingresos +7; huella_consulta +7
RECHAZAR         p=0.905316  pts=448.4  E          tipo_credito_grp -61; puntaje_datacredito -27; plazo_meses -25
RECHAZAR         p=0.012288  pts=640.1  A          politica: sin score de central de riesgo; promedio_ingresos_datacredito +14; ...
RECHAZO_TECNICO  p=None      pts=None   SIN_BANDA  edad_cliente: bajo el minimo (18)
```

### Qué entra en la imagen y qué no

Entra el código de inferencia (4 módulos), el modelo serializado y los artefactos que describen cómo
decidir. No entran los datos de entrenamiento, los notebooks ni las librerías de gráficos: una
imagen de servicio que carga la base de clientes es una fuga de datos esperando ocurrir, y engorda
la imagen sin aportar nada al endpoint.

Como el código localiza la raíz del proyecto buscando `Base_de_datos.csv`, que en la imagen
deliberadamente no existe, `encontrar_raiz()` acepta la variable de entorno `MLOPS_RAIZ`. Fuera del
contenedor la variable no existe y el comportamiento es el de siempre.

El contenedor corre como usuario sin privilegios y expone `data/monitoring` como volumen: ahí queda
el registro de solicitudes y pronósticos, que es la entrada de la Fase 5.

### Lo que el endpoint no tiene

**Sin autenticación y sin rate limiting.** Cualquiera que alcance el puerto 8000 puede pedir
predicciones. En un banco eso sería inaceptable; aquí es una decisión de alcance, y conviene decirlo
en vez de que parezca un olvido.

La razón es que el entregable pide *disponibilizar el modelo*, no construir la plataforma que lo
protege. Autenticación, cuotas por cliente y auditoría de accesos son responsabilidad de la capa que
va delante del servicio (una pasarela de API), no del contenedor del modelo. Meterlas aquí
mezclaría dos problemas y haría la imagen más difícil de desplegar detrás de la infraestructura que
una entidad ya tiene.

Para producción real harían falta tres cosas, en este orden:

| Qué | Por qué |
|---|---|
| OAuth2 o mTLS delante del endpoint | Sin identidad no hay trazabilidad de quién pidió qué |
| Límite de peticiones por cliente | `MAX_SOLICITUDES = 5000` acota el lote, no la frecuencia |
| Cifrado en tránsito | Las 13 columnas son dato semiprivado bajo la Ley 1266 |

El tope de 5.000 solicitudes por lote sí está, y evita que una petición suficientemente grande agote
la memoria del contenedor. Es protección frente a un accidente, no frente a un abuso.

Señalado por `danielCH26` en la revisión de la PR4 y documentado aquí, en la Fase 5, porque es
donde el proyecto pasa de *funciona* a *se puede operar*: lo mismo que empuja a medir la deriva
empuja a declarar qué falta para operar de verdad.


## Monitoreo

El enunciado pide una tabla con los datos pasados al endpoint junto con sus pronósticos, y usarla
con una periodicidad definida para detectar cambios en la población. La tabla ya existía:
`registro_endpoint.csv`, que el despliegue produce en cada llamada. El monitoreo la lee y mide.

### Cómo se mide

**Línea base:** `estratificado_train`, congelada en `data/models/linea_base_monitoreo.json`. Toda
deriva se mide contra esa referencia, nunca contra el periodo anterior. Recalcularla en cada corrida
tendría el mismo problema que recalcular las medianas del saneamiento: la vara de medir se
desplazaría junto con lo medido, y una deriva lenta no se detectaría jamás.

**Los cortes del PSI se importan de la receta.** Si el PSI usara tramos distintos a los del binning,
mediría una cosa y el modelo vería otra. Con los cortes de la receta, un PSI alto significa
literalmente que los tramos que el modelo usa se están llenando distinto.

| Rango de PSI | Lectura |
|---|---|
| Menor a 0,10 | Estable |
| 0,10 a 0,25 | Cambio moderado, vigilar |
| 0,25 o más | Cambio importante, actuar |

### Dos ejes de periodicidad

| Eje | Periodo | Qué responde |
|---|---|---|
| Llegada | Semanal, sobre `momento` | Si lo que entra hoy se parece a lo que el modelo aprendió |
| Cosecha | Mensual, sobre `fecha_prestamo` | Cómo evoluciona el riesgo por *vintage*, que es como trabaja riesgo |

El segundo eje es el que da información en este dataset, porque la Fase 1 midió que la mora va de
1,72% a 9,09% según el mes de desembolso.

**El mínimo se aplica sobre solicitudes distintas, no sobre decisiones.** Una misma solicitud
puntuada dos veces es una observación de la población, no dos. Sin ese detalle, el registro de las
pruebas del endpoint (250 decisiones sobre 50 solicitudes reales) daba un PSI de predicción de
0,2858, es decir *reentrenar*, sobre una población que no había cambiado en absoluto.

### El experimento tiene control

Medir y anunciar que hay deriva no prueba nada si la medición nunca se contrastó. Se evalúan dos
cohortes, ambas pasando **por el endpoint**, no por atajo:

| Cohorte | PSI máximo | Rechazo | Veredicto |
|---|---|---|---|
| Control, muestra de train | 0,0031 | 20,90% | Sin cambio relevante |
| Reciente, `temporal_test` | 0,1509 | 25,07% | Vigilar |

El control sale de la misma población de ajuste, así que debe dar cerca de cero. Da 0,0031. Si no lo
diera, la medición estaría rota y el otro número no merecería crédito. Esa comprobación está dentro
del código: si el control supera 0,10, la ejecución falla.

### Lo que se encontró

Catorce cosechas publicadas de dieciocho. Las cuatro restantes quedan por debajo del mínimo de 100
solicitudes y se declaran como tales en lugar de publicar un número que nadie puede interpretar.

| Cosecha | Solicitudes | PSI máximo | Variable dominante | Rechazo |
|---|---|---|---|---|
| 2024-12 | 248 | 0,259 | `promedio_ingresos_datacredito` | 22,58% |
| 2025-03 | 226 | 0,123 | `puntaje_datacredito` | 21,24% |
| 2025-06 | 129 | 0,240 | `tipo_credito_grp` | 17,05% |
| 2025-08 | 555 | 0,261 | `plazo_meses` | 20,18% |
| 2025-10 | 232 | 0,179 | `plazo_meses` | 29,74% |
| 2025-12 | 174 | 0,470 | `plazo_meses` | 31,61% |
| 2026-01 | 127 | 0,450 | `plazo_meses` | 37,80% |

Hay dos regímenes. Hasta mediados de 2025 la deriva es moderada y la reparten el score y el ingreso
del buró. **Desde 2025-07 `plazo_meses` toma el control y crece hasta 0,470**, y el rechazo sube en
paralelo del 21% previsto al 37,80%.

`plazo_meses` es el coeficiente más alto del modelo (0,8573). Que sea justo esa variable la que se
mueve significa que la entrada más influyente está recibiendo una población distinta de aquella
sobre la que se ajustó. Es la señal accionable: no es ruido repartido, es la variable que más pesa.

### Las cuatro señales, y cuándo llegan

| Señal | Se mide | Estado en la cohorte reciente |
|---|---|---|
| Covariables | De inmediato | PSI máximo 0,1509 en `plazo_meses` |
| Predicción | De inmediato | PSI 0,0166, estable |
| Estabilidad del corte | De inmediato | 25,07% frente al 21,25% previsto |
| Concepto | Solo con etiqueta madura | Ver abajo |

La deriva de predicción sigue estable mientras las covariables ya se movieron. No es contradictorio:
el modelo comprime 17 entradas en un número, y desplazamientos que se compensan entre sí no cambian
la distribución de salida. Por eso se vigilan las cuatro y no una sola.

### La deriva de concepto, con su salvedad

| Subgrupo | n | Mora | AUC-PR | Lift | Plazo medio |
|---|---|---|---|---|---|
| Cohorte completa | 2.154 | 3,20% | 0,0739 | 2,31× | Mixto |
| Solo vencidos | 1.003 | 2,09% | 0,0359 | 1,71× | 6,0 meses |
| Solo jóvenes | 1.151 | 4,17% | 0,0968 | 2,32× | 15,8 meses |

Se reportan los tres, no solo el subgrupo vencido, porque el filtro de madurez está confundido con
el plazo (ver la [advertencia corregida](#partición-de-datos)). El lift acompaña siempre al AUC-PR
porque las tasas base difieren, y el AUC-PR no es comparable entre poblaciones con distinta
proporción de eventos.

### Qué se vigila además

`tendencia_ingresos_reconstruida` tiene 47 casos en train y **cero moras**. Su coeficiente
(-3,9728) no está estimado sino determinado por la separación perfecta, y en el scorecard regala
**115 puntos** sobre un rango de 302. Medido sobre la población completa, la bandera se activa en el
0,54% y voltea 8 decisiones de rechazo a aprobación. El monitoreo publica su tasa de activación
contra la línea base: si sube, el modelo empieza a regalar puntos a más gente por una relación que
nunca se pudo medir.

### Figuras

| Archivo | Qué muestra |
|---|---|
| `psi_por_variable.png` | PSI por variable, control y cohorte reciente lado a lado |
| `evolucion_por_cosecha.png` | PSI máximo y volumen de rechazo a lo largo de las cosechas |


## Resultados

### Insights principales

1. **La mora previa multiplica el riesgo por 8.** De los 55 clientes con saldo en mora previo,
   incumplió el 36,36%; de los 10.708 sin mora previa, incumplió el 4,59%. Son dos tasas
   calculadas por separado dentro de cada grupo (no partes de un total, por eso no suman 100%);
   su cociente da aproximadamente 8.
2. **La ausencia de información del buró es señal de riesgo.** Los 2.930 clientes sobre los que
   la central de riesgo no reporta ingresos incumplen al 5,73%, frente al 4,38% de aquellos con el
   dato disponible (p = 0,0037). El patrón se repite en los clientes sin saldo registrado (7,16%,
   p = 0,031) y sin codeudor registrado (6,78%, p = 0,022).
3. **El tipo de crédito 6 concentra riesgo elevado, pero de magnitud imprecisa.** De sus 21
   créditos, 9 cayeron en mora, frente al 4,75% de la cartera. La asociación es estadísticamente
   sólida (χ², p = 1,77e−13) y su intervalo de confianza al 95% (Wilson) va de **24,5% a 63,5%**,
   muy por encima de la tasa base: el efecto existe. Lo que no se puede afirmar es su tamaño, que
   con 21 observaciones queda entre 5 y 13 veces la base. **Se registra como señal a confirmar con
   más volumen, no como una tasa puntual del 42,86%.**
4. **Depurar el score mejoró su poder predictivo un 78%:** de |r| = 0,068 a |r| = 0,1212, pasando
   a ser el predictor cuantitativo más fuerte.
5. **Los independientes presentan mayor riesgo** que los empleados: 5,51% vs 4,29% (p = 0,0047).
6. **El score discrimina con fuerza, pero no linealmente:** de 64,00% de mora por debajo de 600
   puntos a 2,91% por encima de 850. Su Information Value (0,2136) es el más alto del dataset,
   mientras su correlación de Pearson (0,1212) se leería aisladamente como "relación muy débil".
7. **El plazo del crédito discrimina riesgo:** de 3,91% en créditos de 6–12 meses a 12,66% en los
   de 24–36 meses. El análisis inicial lo había descartado por comparar medianas.
8. **La tasa de mora no es estable en el tiempo:** varía entre 1,72% y 9,09% según el mes de
   desembolso, y el 19,9% de la cartera tiene madurez incompleta.

### Hallazgos que NO resistieron el contraste estadístico

Se registran explícitamente como hipótesis, no como conclusiones:

| Hipótesis | n | Tasa vs. resto | p (Fisher) |
|---|---|---|---|
| Los clientes sin score válido incumplen más | 153 | 6,54% vs 4,72% | 0,3332 |
| Los registros con edad corregida incumplen más | 150 | 7,33% vs 4,71% | 0,1701 |
| Los registros con salario corregido incumplen más | 250 | 6,00% vs 4,72% | 0,3637 |

En los tres casos el intervalo de confianza al 95% contiene la tasa base de comparación. Se
conserva la bandera `sin_historial_crediticio` porque separar esa población **sí** fue necesario
para depurar el score (insight 4), pero su tasa de mora no constituye un hallazgo.

## Conclusiones

- La calidad de los datos exigía intervención antes de cualquier modelado: se corrigieron o
  documentaron **más de 600 registros problemáticos** distribuidos en 6 variables distintas.
- **El diccionario de datos no fue documentación decorativa**: reveló tres errores metodológicos
  reales (correlación de Pearson sobre variable nominal, rango de validación sin fuente, y ceros
  falsos tratados como scores) y produjo dos hallazgos nuevos.
- No existe un predictor individual suficiente. El valor está en la combinación de variables y en
  los atributos derivados, lo que justifica un modelo multivariado en la siguiente fase.
- **La elección de la medida de asociación cambia las conclusiones.** Con un target binario y
  desbalanceado, la correlación de Pearson subestima sistemáticamente el poder discriminante:
  variables descartadas por tener |r| < 0,07 resultan tener poder medio al medirse por tramos. Es
  el aprendizaje metodológico central de esta fase.
- El desbalance de clases (20:1) condiciona el diseño completo del modelado y la elección de
  métricas.

### Recomendaciones de negocio

1. Incorporar tres criterios de alerta temprana en la originación: **mora previa**, **score por
   debajo de 600 puntos** y **tipo de crédito 6**. Los tres tienen efectos grandes, están
   respaldados estadísticamente y son fácilmente explicables ante un cliente o un regulador.
2. Auditar el origen de los datos: los errores hallados apuntan a fallas de captura o exportación
   que conviene corregir en la fuente, no solo en el análisis.
3. Investigar con el área de producto qué representa el tipo de crédito 6.
4. Confirmar la temporalidad de los saldos en mora antes de usarlos en un modelo.

## Limitaciones

| Limitación | Implicación |
|---|---|
| **Sin identificador de cliente** | La unidad de observación es el crédito, no la persona. El 67,9% de las filas comparte perfil demográfico con otra. Impide split agrupado por cliente → riesgo de fuga entre train y test |
| **Fuga de información (confirmada)** | El negocio confirmó que `saldo_mora` y `saldo_mora_codeudor` se registran después del desembolso. Excluidas del modelado desde la Fase 2 |
| **Muestras pequeñas** | Tipo de crédito 6 (21 registros) y mora previa (55 registros): señales indicativas, no concluyentes |
| **Etiqueta no homogénea en el tiempo** | El 19,9% de los créditos vence después del corte de datos: su `Pago_atiempo` refleja el estado observado, no el desenlace final (6,40% vs 4,34%, p = 0,0001) |
| **27% de nulos sin causa estructural** | En `promedio_ingresos_datacredito`, sin patrón que justifique imputación. La bandera de ausencia sí resultó predictor significativo |
| **13 registros de `total_otros_prestamos` > $1.000M** | Sin posibilidad de verificar si son legítimos o errores de captura. Marcados, no imputados (sección 4.1) |
| **Códigos de `tipo_credito` sin confirmar** | No corresponden a ninguna clasificación oficial de la SFC |
| **Sin diccionario de datos original** | La interpretación de varias variables se basa en investigación e inferencia documentada |

## Estructura del repositorio

```
MLOPS_CURSE/
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb            # Ingesta y compuerta de calidad (COMPLETADO)
│   ├── comprension_eda.ipynb         # Fase 1: diccionario, limpieza, EDA (COMPLETADO)
│   ├── ft_engineering.py             # Fase 2: Feature Engineering (COMPLETADO)
│   ├── hueristic_model.py            # Piso de referencia sin modelo (COMPLETADO)
│   ├── model_training.py             # Fase 3: entrenamiento y selección (COMPLETADO)
│   ├── model_evaluation.py           # Fase 3: evaluación y test final (COMPLETADO)
│   ├── model_deploy.py               # Fase 4: motor de inferencia (COMPLETADO)
│   ├── model_monitoring.py           # Fase 5: monitoreo y data drift (COMPLETADO)
│   ├── reglas_negocio.py             # Contrato derivado del EDA (añadido)
│   ├── app.py                        # Fase 4: API del endpoint (añadido)
│   ├── feature_engineering.ipynb     # Narrativa de la Fase 2 (añadido)
│   ├── model_evaluation.ipynb        # Narrativa de la evaluación (añadido)
│   └── hueristic_model.ipynb         # Narrativa del heurístico (añadido)
├── config.json                       # Configuración del proyecto
├── data/
│   └── processed/                    # Salida de la Fase 2
│       ├── estratificado_train.csv        # 8.610 registros (datos particionados)
│       ├── estratificado_test.csv         # 2.153 registros
│       ├── temporal_train.csv             # 8.609 registros (hasta 2025-07-11)
│       ├── temporal_test.csv              # 2.154 registros (desde 2025-07-12)
│       ├── *_features.csv                 # Matrices transformadas (17 features)
│       ├── receta_estratificado.json      # Cortes y WoE (reutilizable en Fase 4)
│       ├── receta_temporal.json
│       ├── reporte_features.json          # Ranking IV, alertas y baseline
│       └── split_metadata.json            # Semilla, tamaños, tasas y exclusiones
│   ├── models/
│   │   ├── modelo_seleccionado.joblib     # El objeto que sirve el endpoint
│   │   ├── pipeline_features_*.joblib     # Pipeline sklearn ajustado
│   │   ├── baseline_heuristico.json       # Piso de referencia
│   │   ├── seleccion_variables.json       # Medición de la etapa 2
│   │   ├── etapa3_logistica.json          # Resultados de la etapa 3
│   │   ├── etapas4y5_comparacion.json     # Rejilla de 3 familias x 3 tratamientos
│   │   ├── evaluacion.json                # Calibración, fairness, umbral y scorecard
│   │   ├── artefacto_despliegue.json      # Contrato de servicio de la Fase 4
│   │   ├── scorecard.csv                  # Tabla de puntos publicada
│   │   ├── linea_base_monitoreo.json      # Referencia congelada de la Fase 5
│   │   ├── monitoreo.json                 # Deriva por cohorte y por cosecha
│   │   └── figuras/                       # Gráficos por partición y de monitoreo
│   └── monitoring/
│       ├── lote_ejemplo.csv               # 50 solicitudes para probar el endpoint
│       └── registro_endpoint.csv          # Entradas y pronósticos (no versionado)
├── Base_de_datos.csv                 # Datos crudos originales (no modificar)
├── Base_de_datos_limpia.csv          # Salida de la Fase 1, entrada de la Fase 2
├── config.json                       # Separador, codificación, semilla y target
├── Dockerfile                        # Fase 4: imagen del endpoint (añadido)
├── docker-compose.yml                # Fase 4: levantar con volumen (añadido)
├── .dockerignore                     # Fase 4: qué no viaja al build (añadido)
├── requirements.txt                  # Dependencias con versiones fijadas
├── requirements-api.txt              # Subconjunto que instala la imagen (añadido)
├── set_up.bat                        # Script de instalación de dependencias
├── .gitignore
└── readme.md
```

Esta estructura es la solicitada en el **Entregable 3**, cuyo enunciado advierte que la estructura de
carpetas **no es modificable** porque el paso a producción se valida con Jenkins. Por eso
`hueristic_model.py` reproduce la errata del enunciado de forma deliberada: corregir la ortografía
sería el error. Por la misma razón se conserva `set_up.bat` y no `setup.bat`: es el nombre que pide
el enunciado.

### Sobre versionar los datos generados

`Base_de_datos_limpia.csv` y el contenido de `data/` son datos generados, y la práctica habitual es
no versionarlos. Aquí se versionan de forma deliberada, por dos razones.

`config.json` declara `Base_de_datos_limpia.csv` como `clean_path`, y es la entrada de
`ft_engineering.py` y de tres notebooks. Sin él, un clon nuevo no puede ejecutar el pipeline hasta
correr las 104 celdas de `comprension_eda.ipynb`.

Y las particiones, las recetas y los modelos de `data/` son lo que permite **auditar los resultados
sin reejecutar nada**: comprobar que las particiones conservan su hash, que las recetas coinciden
con el contrato o que el modelo serializado es el que declara la tabla comparativa.

`PRESENTACION.pptx` sí queda fuera, en `.gitignore`: es un binario que ningún script consume y que
solo ensucia los diffs.

`data/monitoring/registro_endpoint.csv` también queda fuera, y por una razón distinta: crece en cada
llamada al endpoint y lleva marca de tiempo, de modo que no hay dos ejecuciones con el mismo
contenido. Versionarlo produciría un diff distinto cada vez sin ganar auditabilidad. Se reconstruye
con `python mlops_pipeline/model_deploy.py`. `lote_ejemplo.csv` sí se versiona: es una muestra fija,
con semilla, y sirve para probar el endpoint sin abrir la base de clientes.

Los tres archivos marcados como *añadidos* no alteran la estructura exigida, porque añadir no es
modificar. `reglas_negocio.py` publica el contrato del EDA que el resto del pipeline aplica; los dos
notebooks aportan la narrativa de su `.py` correspondiente sin duplicar su lógica.

### Ramas

Se sigue el flujo Gitflow que exige el Stage 1 del Entregable 3:

| Rama | Propósito |
|---|---|
| `master` | Versiones entregadas, con etiqueta de versión |
| `develop` | Integración de las ramas de trabajo |
| `feature1` · `feature2` | Trabajo activo en paralelo |

Los merges hacia `develop` y `master` se hacen con `--no-ff`, de modo que el grafo conserve la forma
del diagrama del enunciado en lugar de aplanarse por *fast-forward*.

## Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.11 | Lenguaje base |
| pandas, numpy | Manipulación y análisis de datos |
| matplotlib, seaborn | Visualización |
| scipy | Estadística (skewness, kurtosis, chi-cuadrado, prueba exacta de Fisher) |
| scikit-learn, imbalanced-learn | Modelado, pipelines y tratamiento del desbalance |
| Jupyter | Notebooks de análisis |
| FastAPI, uvicorn | App y endpoint de predicción por lote |
| Docker | Imagen que contiene librerías, código y artefactos |
| Git / GitHub | Control de versiones (Gitflow, 4 ramas) |

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
jupyter notebook mlops_pipeline/comprension_eda.ipynb
#    (o abrir la carpeta MLOPS_CURSE en VS Code y usar "Run All")
```

**Reproducibilidad:** el notebook se ejecuta de principio a fin sin errores.

**Resolución de rutas.** El notebook está tres niveles por debajo de la raíz del repositorio,
mientras que los CSV viven en la raíz. En lugar de fijar `'../../../'` a mano —frágil ante
cambios de ubicación o de directorio de trabajo del kernel— el notebook localiza la raíz del
proyecto con la función `encontrar_raiz()`, que sube por el árbol de directorios hasta encontrar
`Base_de_datos.csv`. Verificado: funciona desde la raíz del repo, y desde `mlops_pipeline/`. Los scripts de las fases 2 a 5 deben
resolver las rutas de la misma forma.

El archivo crudo nunca se modifica; la salida `Base_de_datos_limpia.csv` se regenera en cada
ejecución completa.

### Levantar el endpoint (Fase 4)

```bash
# 1. Construir los artefactos de despliegue y correr las verificaciones
python mlops_pipeline/model_deploy.py

# 2a. En local, sin contenedor
uvicorn app:app --app-dir mlops_pipeline --port 8000

# 2b. O en contenedor, que es como se entrega
docker compose up --build
```

Documentación interactiva en `http://localhost:8000/docs`.

Probar el endpoint por lote con la muestra versionada:

```bash
curl -X POST http://localhost:8000/predecir/archivo      -F "archivo=@data/monitoring/lote_ejemplo.csv" -o decisiones.csv
```

El paso 1 es obligatorio antes del `docker build`: la imagen copia
`data/models/artefacto_despliegue.json`, que ese script produce. Si falta, el contenedor no arranca,
y eso es deliberado: un contenedor que arranca sin modelo y devuelve errores 500 es peor que uno que
no arranca, porque el primero parece sano.

### Medir la deriva (Fase 5)

```bash
python mlops_pipeline/model_monitoring.py
```

Construye la línea base, hace pasar dos cohortes por el endpoint, lee el registro que esas llamadas
dejaron y lo muestrea por semana de llegada y por mes de cosecha. Falla de forma explícita si el
control supera el umbral de PSI, porque una medición que no discrimina no sirve para decidir nada.

## Fase 2 — Feature Engineering (completada)

Implementada en dos archivos complementarios:

| Archivo | Rol |
|---|---|
| `ft_engineering.py` | **Lógica y única fuente de verdad.** Es el que se ejecutará en producción (Fase 4) |
| `feature_engineering.ipynb` | **Documentación y validación.** Importa el script; no duplica código |

El notebook importa el módulo en lugar de copiar sus funciones: si tuviera su propia copia,
cualquier corrección en el script dejaría de reflejarse y acabaríamos con dos versiones divergentes
de la misma transformación.

**Pasos 1 a 7 completados.**

### Exclusión de variables con fuga de información

El negocio confirmó que `saldo_mora` y `saldo_mora_codeudor` se registran **después** del
desembolso. Son consecuencia y no causa del impago, y el dato no existe al originar el crédito.

| Grupo | Variables | Motivo |
|---|---|---|
| Fuga confirmada | `saldo_mora`, `saldo_mora_codeudor`, `saldo_mora_era_nulo`, `saldo_mora_codeudor_era_nulo` | Registradas tras el desembolso |
| Fuga por precaución | `saldo_total`, `saldo_principal`, `saldo_total_era_nulo`, `saldo_principal_era_nulo` | Mismo corte del buró; IV entre 0,018 y 0,021 (sin coste) |
| No construido | `tiene_mora_previa` | Derivaría de `saldo_mora` |

**El impacto es bajo:** las cinco variables de mayor Information Value están libres de fuga
(`puntaje_datacredito` 0,2136 · `consultas_por_credito` 0,1637 · `huella_consulta` 0,1455 ·
`plazo_meses` 0,1278 · `promedio_ingresos_datacredito` 0,1069). La excluida de mayor peso,
`tiene_mora_previa`, ocupaba el sexto lugar con IV débil (0,0884).

Dataset resultante: **10.763 × 27** (de 35 columnas originales).

### Partición de datos

Se generan **dos particiones** sobre el mismo conjunto, para poder comparar en la Fase 3:

| Partición | Train | Test | Mora train | Mora test | Criterio |
|---|---|---|---|---|---|
| Estratificada | 8.610 | 2.153 | 4,75% | 4,74% | Aleatoria, preservando la proporción de clases |
| Temporal | 8.609 | 2.154 | 5,13% | 3,20% | Corte en 2025-07-12: entrena con el pasado, valida con el futuro |

**Por qué la partición va antes que cualquier transformación.** Si los cortes de binning, los
valores WoE o los parámetros de escalado se calculan sobre el dataset completo, el conjunto de
prueba influye en la transformación y deja de medir el desempeño de forma independiente. Es una
forma silenciosa de fuga que infla las métricas.

**Por qué dos particiones.** La estratificada es el estándar con clases desbalanceadas (20:1) y
evita que el azar deje el test con una proporción de positivos irreal. La temporal reproduce la
situación de despliegue y es necesaria porque la Fase 1 documentó que la tasa de mora **no es
estable entre cohortes** (de 1,72% a 9,09% según el mes).

> ⚠️ **Advertencia sobre la partición temporal.** El conjunto de prueba concentra créditos
> recientes: la madurez incompleta pasa de **11,5% en train a 53,4% en test**, y su tasa de mora
> observada (3,20%) queda por debajo de la de train (5,13%). En la Fase 3 debe evaluarse con y sin
> los registros de madurez incompleta.
>
> **Corregido en la Fase 5.** Se atribuyó esa diferencia al censurado, es decir, a que no ha
> transcurrido tiempo suficiente para observar el impago. Medido, no es lo que ocurre: dentro del
> test temporal los créditos jóvenes tienen **el doble de mora** que los vencidos (4,17% frente a
> 2,09%). La causa es que `madurez_incompleta` está confundida con el plazo, porque un crédito
> sigue vivo justamente por haberse pactado a más meses: los vencidos promedian **6,0 meses** y los
> jóvenes **15,8**. El filtro de madurez no aísla el censurado, selecciona créditos cortos, que son
> estructuralmente menos riesgosos. Ninguno de los dos subgrupos da una lectura sin sesgo, y por eso
> el monitoreo reporta ambos por separado.

**Detalle de implementación.** `fecha_prestamo` incluye hora, por lo que el cuantil caía a mitad de
una jornada y partía el 12 de julio entre ambos conjuntos (1 registro en train, 26 en test). El
corte se normaliza al inicio del día, y una validación explícita rechaza la partición si alguna
fecha calendario aparece en los dos lados.

### Validaciones automáticas

El script aborta con error si alguna falla: suma de particiones distinta al total · índices
compartidos entre train y test · alguna clase ausente · menos de 30 casos de mora en test ·
fechas calendario compartidas en la partición temporal.

### Reproducibilidad

Semilla fija (`random_state: 42` en `config.json`). Ejecuciones sucesivas producen archivos
idénticos, verificado por hash MD5. `split_metadata.json` registra semilla, tamaños, tasas,
fecha de corte y variables excluidas.

### Atributos derivados (paso 4)

Operaciones fila a fila: no dependen de ningún estadístico agregado, por lo que se aplican
idénticamente a train y test sin riesgo de fuga.

`consultas_por_credito` (IV 0,1637 en Fase 1) · `discrepancia_ingresos` (IV 0,0930) ·
`antiguedad_dias` · `mes_prestamo` · `trimestre_prestamo` · `tipo_credito_grp` (tipos 7 y 68,
con 2 y 1 registro, agrupados en "Otros"; el tipo 6 se conserva separado por su riesgo elevado,
con la reserva sobre su magnitud descrita en [Resultados](#resultados)).

`tiene_mora_previa` **no se construye**: derivaría de `saldo_mora`, variable con fuga confirmada.

### Binning y Weight of Evidence (paso 5)

Se eligió **WoE sobre One-Hot** porque el dominio es riesgo crediticio, donde la interpretabilidad
es un requisito regulatorio: hay que justificar la negación de un crédito ante el cliente y ante el
supervisor. Además resuelve los nulos sin imputar — la ausencia de dato se trata como categoría
propia con su WoE estimado a partir de su tasa observada.

**Cortes y valores WoE se ajustan únicamente sobre train.**

**Criterio de fusión de tramos.** Un tramo con pocos registros produce un WoE inestable que el
modelo memoriza como ruido. Caso concreto: la banda de score por debajo de 600 puntos tiene 25
registros y una tasa del 64%, lo que generaría un WoE de +3,57 sobre 16 eventos. El script fusiona
automáticamente los tramos que no alcanzan **5% de la población o 20 eventos de mora**. La categoría
`SIN_DATO` nunca se fusiona: la ausencia de dato no es un valor alto ni bajo.

| `puntaje_datacredito` | WoE | | `plazo_meses` | WoE |
|---|---|---|---|---|
| (280, 700] | **+1,190** | | (0, 6] | −0,019 |
| (700, 750] | +0,544 | | (6, 12] | −0,180 |
| (750, 800] | −0,090 | | (12, 24] | +0,379 |
| (800, 850] | −0,453 | | (24, 90] | **+1,016** |
| (850, 950] | −0,331 | | | |
| **SIN_DATO** | +0,416 | | | |

WoE positivo = más riesgo que el promedio de la cartera. Los 153 clientes sin score se ubican solos
entre las bandas de 700–750 y 750–800, **sin que nadie decida por ellos**: es la validación
retroactiva de la decisión de la Fase 1 de no imputar esos valores.

**No monotonía documentada.** `plazo_meses` no es monótono (0–6 meses tiene más mora que 6–12) y
`puntaje_datacredito` tampoco lo es en el extremo superior. Ambos son hallazgos reales del EDA, no
defectos del binning: no se fuerzan los cortes para producir monotonía artificial.

### Codificación y escalado (paso 6)

Monetarias (`capital_prestado`, `cuota_pactada`, `salario_cliente`, `total_otros_prestamos`):
`log1p` + **escalado robusto** (mediana e IQR) ajustado en train. Se descarta la estandarización
porque la Fase 1 midió asimetrías de 2,2 a 38,5: media y desviación estándar quedan dominadas por
la cola derecha.

Categóricas: WoE sobre sus niveles. Binarias: pasan sin transformar.
**Matriz final: 25 características + target.**

### Validación (paso 7)

**Comparación de IV justa.** El IV crece mecánicamente con el número de tramos, así que comparar la
versión final (4–6 tramos) contra una de 10 la penalizaría sin motivo. Se comparan ambas cosas:

| Variable | IV bruto (10 tramos) | IV final | Retención | Tramos |
|---|---|---|---|---|
| `puntaje_datacredito` | 0,1979 | **0,1898** | 96% | 6 |
| `consultas_por_credito` | 0,1344 | **0,1148** | 85% | 5 |
| `huella_consulta` | 0,1470 | 0,1129 | 77% | 5 |
| `promedio_ingresos_datacredito` | 0,1133 | **0,1044** | 92% | 6 |
| `plazo_meses` | 0,1279 | 0,0904 | 71% | 4 |
| `discrepancia_ingresos` | 0,0862 | 0,0788 | 91% | 6 |

**Retención media del 89% usando menos de la mitad de los tramos.** El IV que se pierde es
precisamente el sobreajuste que la fusión elimina.

**Baseline comparativo.** Regresión logística con validación cruzada estratificada de 5 pliegues,
medida en **AUC-PR** sobre la clase minoritaria (con 4,75% de eventos, el AUC-ROC es demasiado
optimista y la exactitud inservible). El escalado va dentro del pipeline para ajustarse en cada
pliegue de entrenamiento y no en el de validación.

| Partición | Originales | Transformadas | Ganancia | Lift vs azar |
|---|---|---|---|---|
| Estratificada | 0,1197 | **0,1462** | +22,1% | 3,08x |
| Temporal | 0,1336 | **0,1617** | +21,0% | 3,15x |

Las características transformadas superan a las originales en ambas particiones, lo que confirma
que el trabajo de esta fase aporta información y no solo reordena la existente.

**Alertas registradas.** `cant_creditosvigentes` (IV 0,019) y `tipo_laboral` (IV 0,016) quedan por
debajo del umbral de 0,02. Se conservan porque una variable débil puede aportar en combinación
dentro de un modelo multivariado, pero son las primeras candidatas a descarte en la Fase 3.
Ninguna variable supera IV 0,5, umbral que habría obligado a investigar fuga.

### Verificación de ausencia de fuga

Prueba específica: se alteró radicalmente el conjunto de prueba y se comprobó que **la receta
aprendida no cambia**, y que todos los valores WoE aplicados a test provienen del mapa de train.
Los niveles no vistos en entrenamiento reciben WoE = 0 (riesgo promedio), la decisión conservadora.

### Artefactos generados

`receta_estratificado.json` y `receta_temporal.json` contienen cortes, fusiones y valores WoE.
Son el artefacto reutilizable: en la Fase 4 el modelo en producción debe aplicar **exactamente**
estas transformaciones, sin recalcularlas sobre datos nuevos.

### Correcciones P1–P4 (auditoría de cierre)

Antes de dar la fase por cerrada se auditó el pipeline completo y se corrigieron cuatro problemas.

#### P1 — Variables de la ventana de observación

`antiguedad_dias` y `madurez_incompleta` **no describen al cliente sino cuánto tiempo lleva vivo el
crédito**. Tres problemas medidos:

1. En la partición temporal sus rangos **no se solapan** entre train y test (train [288, 516] días,
   test [0, 288]): la variable reproduce el propio criterio de partición.
2. Correlacionan con el target a través del sesgo de madurez, no del perfil del cliente.
3. En originación valdrían ~0 y 1 para todos los créditos — fuera del rango de entrenamiento
   (*train-serving skew*).

**Corrección:** salen de la matriz y quedan como columnas auxiliares. Se añade la constante
`MODO_USO = "originacion"` que documenta el supuesto. `trimestre_prestamo` se elimina por redundancia
y `mes_prestamo` pasa a tratarse como categoría con WoE (como entero, un modelo lineal asumiría que
diciembre "vale" doce veces enero).

#### P2 — Colinealidad

`sin_historial_crediticio` y `promedio_ingresos_datacredito_era_nulo` marcaban **exactamente las
mismas filas** que el tramo `SIN_DATO` del WoE de su variable de origen (correlación medida
**1,0000**). No aportaban información nueva y creaban colinealidad perfecta, que en una regresión
logística vuelve inestables los coeficientes y puede invertir signos.

**Corrección:** se retiran; la versión WoE es preferible porque conserva la magnitud del riesgo
(+0,4174 y +0,2742) en lugar de un 0/1. Se añade la función `detectar_colinealidad()`, que combina
correlación por pares y **VIF** (detecta que una columna sea combinación lineal de varias, cosa que
la correlación por pares no ve).

**Resultado:** 0 pares con |r| ≥ 0,80 y VIF máximo 3,01 (umbral de alerta: 10).

#### P3 — Binning de `plazo_meses`

Con los cortes anteriores el tramo (12,18] tenía 275 registros y 12 eventos, por debajo de ambos
mínimos, y la fusión lo unía con (18,24] — justo donde empieza la señal (8,16% de mora). El IV de la
variable **bajaba** al transformarla (0,1069 → 0,0904).

**Corrección:** cortes revisados a `[0, 6, 18, 24, 90]`. La Fase 1 muestra que (12,18] (4,36%) se
parece mucho más a (6,12] (3,91%) que a (18,24] (8,16%), así que se agrupan desde el corte inicial.

**Resultado:** IV 0,0904 → **0,0981**. El gradiente del tramo largo queda preservado:

| Tramo | WoE |
|---|---|
| (0, 6] | −0,0185 |
| (6, 18] | −0,1747 |
| (18, 24] | +0,5791 |
| (24, 90] | +1,0157 |

*Honestidad sobre el alcance:* `vs_comparable` sigue siendo ligeramente negativo (−0,0088). La
mejora es parcial: se prefieren cortes con significado de negocio sobre cuantiles ciegos.

#### P4 — Monotonía

El WoE de `puntaje_datacredito` no era monótono en el extremo alto: (800,850] daba −0,4526 y
(850,950] daba −0,3305. Eso obliga a decirle a un supervisor que un cliente con mejor score es más
riesgoso que otro con peor score.

**Corrección:** se declara `VARS_MONOTONAS = {"puntaje_datacredito": "descendente"}` y se aplica
`_forzar_monotonia()`, que fusiona tramos contiguos que violan la dirección esperada (idea del
algoritmo PAVA). `SIN_DATO` no participa: no ocupa posición en la escala de riesgo.

**No se aplica a `plazo_meses`** (la Fase 1 mostró que 0–6 meses tiene más mora que 6–12: la
relación no es monótona y forzarla destruiría información real) **ni a `edad_cliente`** (no hay
razón de negocio para exigirlo).

**Resultado:** monotonía restaurada con **IV prácticamente idéntico** (0,1898 → 0,1899), lo que
confirma que la inversión era ruido muestral y no señal.

| Tramo | WoE |
|---|---|
| (280, 700] | +1,1914 |
| (700, 750] | +0,5455 |
| (750, 800] | −0,0890 |
| (800, 950] | −0,4191 |
| SIN_DATO | +0,4174 |

#### Efecto conjunto de las correcciones

| | Antes | Después |
|---|---|---|
| Características | 25 | **19** |
| AUC-PR estratificada | 0,1462 | 0,1424 (−2,6%) |
| Ganancia vs. originales (estratificada) | +22,1% | +18,5% |
| AUC-PR temporal | 0,1617 | **0,1748 (+8,1%)** |
| Ganancia vs. originales (temporal) | +21,0% | **+29,9%** |
| Lift temporal sobre el azar | 3,15× | **3,41×** |
| VIF máximo | no medido | 3,01 |

La caída de 2,6% en la partición estratificada es el **coste honesto** de retirar variables que no
estarán disponibles en producción. La mejora del 8,1% en la temporal indica que esas variables
estaban perjudicando la generalización justo donde importa: al predecir sobre créditos nuevos.

### Estado

**Las cinco fases completadas y auditadas.** Dataset final de 17 características, sin fuga, sin
colinealidad y con monotonía verificada; modelo elegido sobre validación cruzada, evaluado una sola
vez sobre test, auditado en calibración y *fairness*, traducido a scorecard, servido en un endpoint
por lote dentro de una imagen Docker, y monitoreado por cosecha sobre el registro que ese endpoint
produce.

**Lo que el monitoreo dejó sobre la mesa.** Desde la cosecha de 2025-07, `plazo_meses` domina la
deriva y llega a un PSI de 0,470, mientras el rechazo sube del 21% previsto al 37,80%. Es el
coeficiente más alto del modelo, de modo que la entrada más influyente está recibiendo una población
distinta de aquella sobre la que se ajustó. Con la cartera en ese estado, la decisión que sigue es
reentrenar sobre datos recientes y volver a fijar el punto de operación, no ajustar el umbral a mano
sobre la población que se quiere medir.

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
