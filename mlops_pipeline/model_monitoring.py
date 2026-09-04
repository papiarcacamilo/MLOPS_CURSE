"""
Monitoreo y medida de data drift
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : el registro de model_deploy.py (entradas + pronosticos del endpoint)
Salida   : metricas de deriva con periodicidad definida
Estado   : PENDIENTE -- se implementa al final del flujo

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Crea el trabajo de monitoreo que trae en una tabla los datos pasados al
     endpoint junto con los pronosticos entregados por este y los utiliza, con
     una periodicidad definida, para muestrear y obtener metricas que permitan
     detectar cambios en la poblacion que puedan afectar el desempenio del
     modelo. Medida del Data drift."

POR QUE ESTE PROYECTO TIENE UN CASO DE DERIVA REAL

No es un ejercicio teorico. La Fase 1 documento que la tasa de mora de esta
cartera oscila entre 1.72% y 9.09% segun el mes de desembolso. Un modelo
entrenado sobre la mezcla completa vera poblaciones que se desvian de esa media
con regularidad, y la particion temporal ya lo confirmo: entrenando con el
pasado y validando con el futuro, la estructura cambia.

QUE VIGILAR

    Deriva de covariables   distribucion de cada feature contra su linea base
                            de entrenamiento. PSI por tramo es lo natural aqui:
                            reutiliza los mismos cortes del binning de la Fase 2
    Deriva de prediccion    distribucion de las probabilidades emitidas. Se
                            detecta sin esperar la etiqueta, y por eso es la
                            senial mas temprana
    Deriva de concepto      relacion entre features y target. Requiere que la
                            etiqueta madure, asi que llega tarde por definicion
    Estabilidad del corte   si el punto de operacion sigue rechazando el ~21%
                            que rechazaba, o la poblacion se ha movido

LA TRAMPA DE ESTE DATASET: LA ETIQUETA MADURA TARDE

El 19.9% de los creditos tiene madurez incompleta; en el test temporal, el
53.4%. Un credito recien desembolsado aparece como "al dia" simplemente porque
no ha transcurrido el plazo. Medir desempenio sobre creditos jovenes subestima
la mora de forma sistematica.

Consecuencia practica: la deriva de covariables y de prediccion se puede medir
de inmediato; la de concepto solo sobre cohortes ya vencidas. Confundirlas
produce falsas alarmas de mejora.

LINEA BASE

Se toma de data/processed/*_train.csv y de las recetas: son la fotografia de la
poblacion sobre la que el modelo aprendio. Cualquier medida de deriva es contra
esa referencia, no contra el periodo anterior.
================================================================================
"""
