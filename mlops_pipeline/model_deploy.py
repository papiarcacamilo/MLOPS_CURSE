"""
Despliegue del modelo
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : el mejor modelo + receta_*.json + reglas_negocio.py
Salida   : endpoint para prediccion por lote, empaquetado en una imagen
Estado   : PENDIENTE -- se implementa tras la evaluacion

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Se toma el mejor modelo desplegado y una imagen que contenga las librerias
     y el codigo para una app que permita disponibilizar dicho objeto y despliega
     el modelo en un endpoint que puede utilizarse para predicciones (por batch)."

Son tres piezas: la app, la imagen que la contiene, y el endpoint por lote.

EL REQUISITO QUE NO PUEDE FALLAR

Un cliente nuevo debe atravesar EXACTAMENTE las mismas transformaciones que el
conjunto de entrenamiento. Si el endpoint recalcula cortes, reajusta WoE o
imputa de otra forma, el modelo recibe una entrada que no se parece a lo que
aprendio -- train-serving skew -- y falla en silencio: sigue devolviendo
probabilidades, solo que equivocadas.

Por eso las recetas de la Fase 2 se guardaron como artefacto. El endpoint las
APLICA; nunca las reajusta.

ORDEN DE LA CADENA DE INFERENCIA

    1. Validar        validar_dataframe() del contrato. Un registro que
                      incumple el contrato se rechaza, no se puntua. Ojo con las
                      fechas: llegan en el formato de origen d/m/Y, y por eso el
                      contrato declara la convencion en vez de dejar que pandas
                      la adivine
    2. Transformar    aplicar receta_*.json: derivadas, binning, WoE, escalado
    3. Puntuar        el modelo serializado devuelve la probabilidad de mora
    4. Decidir        aplicar el umbral de operacion elegido en la evaluacion
    5. Registrar      guardar entrada y prediccion: es el insumo de
                      model_monitoring.py, que sin esta tabla no puede medir nada

SCORECARD (etapa 10)

La tabla de puntos interpretable se construye aqui, no en el entrenamiento: es
un artefacto de despliegue. Convierte los coeficientes sobre WoE en puntos
enteros por tramo, que es el formato con el que un analista de riesgo justifica
una negacion ante el cliente y ante el supervisor.

En credito la interpretabilidad no es un empate tecnico, es una ventaja: ante
una ganancia marginal de AUC-PR, gana el modelo simple.
================================================================================
"""
