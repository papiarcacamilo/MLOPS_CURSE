"""
Entrenamiento y seleccion de modelos
================================================================================
Proyecto : Modelo de riesgo crediticio (MLOPS_CURSE)
Entrada  : data/processed/*_train_features.csv + receta_*.json
Salida   : el objeto del modelo seleccionado como mejor
Estado   : PENDIENTE -- se implementa tras medir la seleccion de variables

QUE PIDE LA ENTREGA 3, LITERALMENTE

    "Se entrenan y evaluan diferentes modelos. De este debe resultar el objeto
     del modelo seleccionado como el mejor (model performance, consistency,
     scalability). Se deben utilizar las funciones: summarize_classification y
     build_model. Utilizar graficos comparativos para los modelos principales.
     Tabla resumen."

Los dos nombres de funcion son obligatorios. Los tres criterios de seleccion
vienen del diagrama de Venn del enunciado y conviene que sean columnas
explicitas de la tabla resumen, no una mencion de paso:

    performance   AUC-PR sobre validacion cruzada
    consistency   desviacion entre folds; un modelo inestable no es desplegable
    scalability   coste de entrenamiento e inferencia, y de mantenerlo vivo

ETAPAS QUE CUBRE

    2  Seleccion de variables      19 / 18 / 17 features con los mismos folds
    3  Logistica sobre WoE         referencia; todo coeficiente debe salir
                                   positivo, o el WoE esta invertido
    4  Arboles y boosting          RandomForest e HistGradientBoosting
    5  Tratamiento del desbalance  sin tratar / class_weight / SMOTE / umbral

REGLA QUE NO SE PUEDE ROMPER

El conjunto de prueba NO se toca aqui. Toda la seleccion se hace con validacion
cruzada sobre train, con StratifiedKFold(5, shuffle=True, random_state=42) --
los mismos folds que uso la sonda de la Fase 2, para que las cifras sean
comparables contra su AUC-PR de 0.1424 (estratificado) y 0.1748 (temporal).

El test se abre una sola vez, en model_evaluation.py.

QUE HAY QUE SUPERAR

    Piso heuristico     AUC-PR 0.0753 estratificado / 0.0862 temporal
    Sonda de la Fase 2  AUC-PR 0.1424 / 0.1748

El piso dice si el modelo aporta sobre no tener modelo. La sonda dice si
aprovecha la informacion que las features ya contienen. Un modelo que supere el
primero pero no se acerque al segundo no esta aprovechando lo que tiene.
================================================================================
"""
