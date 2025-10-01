# Prediccion de Morosidad en Educacion Superior

Este repositorio contiene el flujo completo para construir y desplegar un modelo de clasificacion que estima la probabilidad de impago de nuevos alumnos en una escuela de negocios. Incluye el preprocesamiento de datos historicos, el entrenamiento del modelo y una aplicacion Streamlit para consultar predicciones de forma interactiva.

## Caracteristicas principales
- Limpieza, normalizacion y enriquecimiento de los datos historicos (`scr/data.py`).
- Utilidades de analisis estadistico y generacion de features (`scr/utils.py`, `scr/feature.py`).
- Modelo Random Forest optimizado serializado en `models/modelo_random_forest_optimizado.pkl` y pipeline completo en `models/pipeline_final.pkl`.
- Aplicacion web (`streamlit_app/app.py`) que calcula automaticamente los campos dependientes y muestra la probabilidad de impago solo cuando supera el 30 %.
- Notebooks exploratorios y de entrenamiento en la carpeta `notebooks/` para reproducir el proceso completo.

## Estructura del proyecto
- `data/`
  - `raw/`: fuentes originales (Excel)
  - `processed/`: datos procesados listos para modelar
  - `test/`: conjuntos de prueba exportados desde los notebooks
- `models/`: artefactos entrenados (`.pkl`)
- `notebooks/`: analisis exploratorio y entrenamiento
- `scr/`
  - `data.py`: pipeline de tratamiento de datos
  - `feature.py`: conversores y helpers
  - `utils.py`: utilidades de estadistica y evaluacion
- `streamlit_app/`
  - `app.py`: aplicacion Streamlit de consulta del modelo
- `requirements.txt`: dependencias del proyecto
- `README.md`

## Requisitos
- Python 3.10+ (el proyecto se ha probado con Python 3.11).
- Dependencias listadas en `requirements.txt`:
  - joblib, numpy, openpyxl, pandas, scikit-learn, scipy, streamlit.

Instalacion:
```
python -m venv .venv
. .venv\Scripts\activate     # En Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso de la aplicacion Streamlit
1. Activa tu entorno virtual y asegurate de haber instalado las dependencias.
2. Ejecuta el servidor local:
   ```
   streamlit run streamlit_app/app.py
   ```
3. Abre el enlace que Streamlit muestra en consola (por defecto http://localhost:8501).

La interfaz solicita los datos relevantes del alumno; el script calcula automaticamente:
- Importe de inscripcion (facturacion neta x porcentaje de inscripcion).
- Importe pendiente (facturacion neta - importe cobrado, nunca negativo).
- Consistencia de los medios de pago, que alimenta features binarias.

Si el importe pendiente es cero, la probabilidad de impago se fuerza a 0. Solo se muestra la probabilidad cuando es mayor o igual al 30 %.

## Reentrenamiento del modelo
- El preprocesamiento de datos se encuentra en `scr/data.py`. Ejecutalo cuando existan nuevas fuentes en `data/raw/` para actualizar `data/processed/data.csv`.
- Los notebooks de la carpeta `notebooks/` documentan el entrenamiento, la seleccion de hiperparametros y la evaluacion. Ejecutalos en orden para regenerar el modelo si se actualizan los datos.
- Guarda los artefactos nuevos en `models/` y actualiza la app si cambia el nombre o formato del modelo.

## Notas sobre los datos
Los archivos en `data/raw/` contienen informacion sensible de alumnos/clientes, por lo que **no se deben versionar ni compartir** fuera del entorno controlado. El repositorio distribuye unicamente scripts y artefactos anonimizados.

## Conclusiones del EDA
- Los graficos `reports/01_distribucion_impago_segun_pi.jpg` y `reports/02_distribucion_impago_segun_pi_100.jpg` muestran que el 18.9 % de los alumnos termina en morosidad, con grandes diferencias segun el medio de pago inicial: `TARJETA` reduce la tasa al 12.6 %, mientras `PAYBAY` y `PAYCOMET` la elevan por encima del 23 %.
- `reports/03_distribucion_impago_segun_pi_agrupado.jpg` y `reports/04_distribucion_impago_segun_pi_agrupado_2.jpg` confirman que mantener el mismo medio de pago en PI e importe pendiente baja la morosidad del 22.0 % al 13.9 %.
- El heatmap de `reports/05_correlacion_categoricas_morosidad.jpg` resalta la importancia de las variables ligadas a los metodos de cobro (`MANTIENE MEDIO PAGO`, `MEDIO PAGO IMPORTE PENDIENTE`), alineadas con las diferencias observadas en las distribuciones anteriores.
- `reports/06_correlacion_numericas_morosidad.jpg` evidencia la fuerte relacion negativa entre `Importe Cobrado` y morosidad (r = -0.55), seguida por correlaciones mucho menores en `% INSCRIPCION` y `% DTO`.
- Los boxplots de `reports/07_boxplots.png` muestran que los alumnos al corriente de pago presentan medianas de `Importe Cobrado` cercanas a 4.9 k€, casi el doble que los morosos (2.6 k€), reforzando la relevancia de los importes abonados tempranamente.

## Conclusiones de la evaluacion del modelo
- Sobre un conjunto de prueba estratificado (20 %), el pipeline (`models/pipeline_final.pkl`) alcanza `accuracy = 0.63`, `recall = 0.64`, `precision = 0.29`, `f1 = 0.39` y `roc_auc = 0.67`; estos resultados se reflejan en `reports/11-matrices_confusion_final.jpg`.
- `reports/12-curvas_ROC.jpg` muestra un ROC estable alrededor de 0.67 y `reports/14-curvas_precision_recall.jpg` evidencia que operar en recalls por encima de 0.6 implica sacrificar precision, coherente con el uso del modelo como filtro temprano.
- El comparativo `reports/13-comparacion_auc_f1.jpg` confirma que el Random Forest optimizado supera a los modelos de referencia en AUC y F1, mientras que `reports/09-matrices_confusion_modelos_optimizados.jpg` ilustra la mejora frente a configuraciones previas.
- Las visualizaciones del arbol (`reports/08-decision_tree_structure.jpg` y `reports/10-decision_tree_structure.jpg`) muestran divisiones iniciales basadas en `Importe Cobrado`, `MANTIENE MEDIO PAGO` y el medio de cobro, en linea con los hallazgos del EDA.

## Contacto
Para dudas o contribuciones, abre un issue o contacta al mantenedor del repositorio.
