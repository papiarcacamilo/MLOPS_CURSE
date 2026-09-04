"""
Evaluacion del modelo desplegado
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : el modelo seleccionado + data/processed/*_test*.csv
Salida   : pestania de metricas de desempenio
Estado   : PENDIENTE. Se implementa despues de model_training.py

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Genera un proceso de evaluacion que crea una pestania de metricas para
     conocer el desempenio del modelo desplegado."

ETAPAS QUE CUBRE

    6  Prueba de estres temporal   con y sin registros de madurez incompleta
    7  Calibracion                 curva + Brier; Platt o isotonica si desvia
    8  Fairness                    tasas de error por grupo
    9  Evaluacion final en test    UNA SOLA VEZ, con el modelo ya elegido

METRICAS Y POR QUE ESTAS

    AUC-PR      PRINCIPAL. Con 4.75% de eventos es la que refleja el desempenio
                real. La exactitud queda descartada: un modelo que apruebe todo
                acierta el 95.25% sin aportar nada
    KS          separacion entre distribuciones; sobre 0.30 se considera util
    Gini        2*AUC-1, estandar regulatorio comparable entre entidades
    Recall @ k% traduce el modelo a decision operativa
    Brier       calidad de las probabilidades, no solo del orden

DISCRIMINACION Y CALIBRACION SON PROPIEDADES INDEPENDIENTES

Un modelo puede ordenar perfecto y estar pesimo calibrado, y aqui es esperable:
class_weight="balanced" distorsiona las probabilidades por diseno. Importa
porque la probabilidad de incumplimiento entra al calculo de provisiones: mal
calibrado significa provisionar mal, que es un problema contable antes que
estadistico.

LA PRUEBA DE ESTRES TEMPORAL

Si un modelo va bien en la particion estratificada y mal en la temporal,
aprendio el periodo, no el riesgo. Hay que evaluar por separado excluyendo los
registros de madurez incompleta: en el test temporal son el 53.4%, y su
Pago_atiempo observado subestima el impago real porque el credito aun no ha
vencido.

FAIRNESS

Ejes disponibles: edad_cliente y tipo_laboral. Dos puntos que deben poder
defenderse:

  1. tipo_laboral tiene IV ~0.015 (poder casi nulo) y es eje de
     discriminacion. Se asumiria riesgo etico a cambio de nada medible.
  2. Quitar la variable NO elimina el sesgo. woe_promedio_ingresos_datacredito
     distingue a clientes con menor huella financiera formal, lo que
     correlaciona con informalidad laboral y puede actuar como proxy. Debe
     medirse, no asumirse: tipo_laboral se conserva como eje de auditoria
     aunque no entre al modelo.

Se miden tasa de aprobacion y tasa de error POR GRUPO, no solo AUC global.

DECLARACION OBLIGATORIA AL REPORTAR

El dataset no tiene identificador de cliente. La unidad de observacion es el
credito, no la persona, y el 67.9% de las filas comparte perfil demografico con
otra. No se pudo aplicar particion agrupada por cliente. Debe constar junto a
cualquier metrica que se publique.
================================================================================
"""
