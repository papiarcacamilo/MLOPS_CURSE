# Comparación de nubes: AWS, Azure y GCP

Entregable 5, Fase 3. Compara las tres nubes en los 14 criterios del enunciado, a partir de lo que
se construyó en las Fases 1 y 2: entrenar con AutoML, entrenar con un pipeline propio y operar el
ciclo con CI/CD.

**Método.** GCP es la nube que se utilizó en el proyecto. AWS y Azure se comparan a través de su
documentación oficial.

---

## 1. Experiencia medida en GCP

Todo ocurrió en la cuenta propia, en `us-central1`.

### 1.1 Qué hubo que configurar

| Paso | Para qué | Detalle |
|---|---|---|
| Dos proyectos | Separar datos y entrenamiento | `nb-ret-datos-dev` para Retail, dueño del dato; `nb-rsk-scoring-dev` para Riesgo, que entrena |
| Activar Vertex AI | Usar AutoML, pipelines y registro | Una API en el proyecto de Riesgo |
| Datos en BigQuery | Fuente de los entrenamientos | Tabla original en Retail; tablas de modelado en el dataset `modelado` de Riesgo |
| Bucket | Paquetes de código y artefactos de los pipelines | `gs://nb-rsk-scoring-dev-vertex`, sin acceso público |
| Cuenta de servicio | Correr el pipeline con permisos mínimos | `sa-entrenamiento`, sin llaves descargadas |
| Permisos | Dar a cada identidad solo lo necesario | 8 asignaciones: 5 para la cuenta de servicio y 3 para el agente de Vertex AI |
| Presupuesto | Vigilar el gasto | USD 25 con alertas al 50, 80 y 100% |

### 1.2 Dificultades

| Problema | Causa | Cómo se resolvió |
|---|---|---|
| El agente de servicio de Vertex AI no tenía ningún rol en el proyecto | El proyecto no tenía roles para ningún agente de servicio; la causa no se investigó | Se le asignó `aiplatform.serviceAgent` antes de lanzar AutoML |
| AutoML podía descartar filas con nulos | Las columnas numéricas no admiten valores inválidos por defecto | Se declararon con `invalidValuesAllowed` |
| La consola mostraba AUC-PR 0,967 | Promedia las dos clases, y domina la que paga | Se midió la clase mora con nuestra propia función: 0,1952 |
| La primera corrida del pipeline falló | Kubeflow entra al contenedor sin sus dependencias y faltaba `google-cloud-storage` | Se fijó la librería y se agregó un test |
| El ejecutor local de Kubeflow no funciona en Windows | Necesita `sh` | Se probó cada etapa en entornos limpios con lo mismo que instala el YAML |
| El costo no se pudo leer por API | La API de facturación no estaba activa | Se consultó en la consola |

### 1.3 Interfaz gráfica

| Pantalla | Para qué se usó | Observación |
|---|---|---|
| Canalizaciones de entrenamiento | Ver la corrida de AutoML | Muestra estado, duración y configuración |
| Registro de modelos, pestaña Evaluar | Métricas, matriz de confusión e importancia de variables | El valor destacado promedia las clases; el de la mora aparece al elegir la etiqueta `1` |
| Canalizaciones, ejecuciones | Ver el pipeline propio como un diagrama | Cada etapa cambia de color al terminar |
| Facturación, Informes y Créditos | Costo real y saldo | El cargo puede tardar hasta un día en aparecer |

### 1.4 Tiempos medidos

| Actividad | Tiempo | Cómo se midió |
|---|---|---|
| Implementar AutoML: tabla, script, permisos y lanzamiento | Cerca de 1 h 40 min | Desde el commit anterior hasta el lanzamiento; incluye documentación |
| Cola de AutoML hasta empezar | 3 min | Estado `PENDING` a `RUNNING` |
| Entrenamiento de AutoML | 2 h 9 min 56 s | Inicio y fin de la corrida |
| Implementar el pipeline propio, con la falla y su corrección | Cerca de 2 h 20 min | Desde el commit anterior hasta el fin de la corrida exitosa; incluye documentación |
| Corrida completa del pipeline | 12 min 53 s | Inicio y fin de la corrida |
| Cómputo real dentro del pipeline | 4 min 52 s | Suma de los 5 trabajos: 18 s, 50 s, 40 s, 70 s y 114 s |
| Entrenar la misma receta en local | 1,2 s | Promedio de tres repeticiones |
| CI en cada push | Cerca de 2 min | GitHub Actions: tests y SonarCloud |

**Lectura:** en un pipeline administrado, preparar las máquinas pesa más que el cómputo. De los
12 min 53 s, solo 4 min 52 s fueron trabajo real; el resto fue aprovisionar e instalar paquetes.

### 1.5 Uso de recursos

| Trabajo | Máquina | GPU |
|---|---|---|
| AutoML | No se ve: Google la elige; se controla y se cobra en nodo-horas | No visible |
| Cada etapa del pipeline | `e2-standard-4`, la máquina por defecto | Ninguna |
| Entrenamiento local | El portátil del autor | Ninguna |

Ningún trabajo necesitó GPU: son 10.763 filas tabulares y un modelo lineal.

### 1.6 Costos medidos

| Concepto | Costo | Evidencia |
|---|---|---|
| AutoML, 1 nodo-hora | USD 21,25 | Facturación, noviembre |
| Pipeline propio, dos corridas, incluida la fallida | USD 0,07 | Facturación, 7 de noviembre |
| BigQuery y Cloud Storage | Sin cargo visible | Dentro del nivel gratuito |
| Endpoints | Ninguno creado | Comprobado en Vertex AI |
| **Total de la fase en GCP** | **USD 21,32** | Cubierto por el crédito educativo, que quedó en USD 7,42 de USD 50 |

**Lectura:** AutoML costó unas 300 veces más que el pipeline propio, por un modelo mejor en AUC-PR
pero sin interpretabilidad.

### 1.7 CI/CD, versionado y gobernanza que se usaron

| Capacidad | Cómo se resolvió |
|---|---|
| Integración continua | GitHub Actions y SonarCloud en cada push; Jenkins después del merge |
| Pipeline como código | Kubeflow en Python, compilado en el CI en cada push |
| Trazabilidad | La corrida del pipeline y su modelo llevan el commit en las etiquetas, y el código viaja como artefacto |
| Versionado de modelos | Model Registry: `automl-mora-e5` y `python-woe-pipeline-e5`, con etiquetas |
| Compuerta de calidad | El pipeline solo registra si el AUC-PR supera el mínimo de Jenkins |
| Control de acceso | Cuenta de servicio con mínimo privilegio y permisos por dataset |
| Control de gasto | Presupuesto con alertas y un tope de nodo-horas en AutoML |

### 1.8 Lo que no se midió

- **Despliegue en la nube:** no se creó un endpoint, porque cobra por hora. El modelo se sirve con la
  imagen de FastAPI del proyecto.
- **Monitoreo administrado de Vertex AI:** el monitoreo de deriva del proyecto es propio, del
  Entregable 3.
- **Registros de auditoría:** Cloud Logging se usó para depurar, pero no se revisaron los registros de
  auditoría de permisos.
- **AWS y Azure:** no se usaron. Todo lo suyo será documentado o estimado.
