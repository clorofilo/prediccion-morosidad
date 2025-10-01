# Prediccion de Morosidad en Educacion Superior

Este repositorio contiene el flujo completo para construir y desplegar un modelo de clasificacion que estima la probabilidad de impago de nuevos alumnos en una escuela de negocios. Incluye el preprocesamiento de datos historicos, el entrenamiento del modelo y una aplicacion Streamlit para consultar predicciones de forma interactiva.

## Caracteristicas principales
- Limpieza, normalizacion y enriquecimiento de los datos historicos (`scr/data.py`).
- Utilidades de analisis estadistico y generacion de features (`scr/utils.py`, `scr/feature.py`).
- Modelo Random Forest optimizado serializado en `models/modelo_random_forest_optimizado.pkl`.
- Aplicacion web (`streamlit_app/app.py`) que calcula automaticamente los campos dependientes y muestra la probabilidad de impago.
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

## Contacto
Para dudas o contribuciones, abre un issue o contacta al mantenedor del repositorio.
