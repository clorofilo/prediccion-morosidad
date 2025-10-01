import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pipeline_final.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "data.csv"

TARGET_COLUMN = "Moroso"
PERCENT_COLUMNS = ["% INSCRIPCION", "% DTO"]
DERIVED_BOOL_COLUMNS = ["MANTIENE MEDIO PAGO", "DIFERENCIA PI vs. IMPORTE PTE"]
DROP_COLUMNS = [
    "FECHA PRODUCCIÓN", "FECHA 1ra CUOTA", "FECHA 1ra CUOTA ORIGINAL",
    "% Impagado Actual Vdo", "Importe Impagado Actual", "ID PROGRAMA",
    "PRECIO CURSO", "AGRUPACION NACIONALIDAD", "AGRUPACION PAÍS DE RESIDENCIA"
]

# Columnas clave
COL_DIA_CERO = "DIA CERO"
COL_FACTURACION = "FACTURACIÓN NETA"
COL_IMPORTE_INSCRIPCION = "IMPORTE INSCRIPCIÓN"
COL_PCT_INSCRIPCION = "% INSCRIPCION"
COL_MEDIO_PAGO_PI = "MEDIO PAGO PI"
COL_IMPORTE_PENDIENTE = "IMPORTE PENDIENTE PAGO"
COL_MEDIO_PAGO_PTE = "MEDIO PAGO IMPORTE PENDIENTE"
COL_NUM_CUOTAS = "NUMERO DE CUOTAS"
COL_FORMA_PAGO = "FORMA DE PAGO"
COL_ASESOR = "ASESOR"
COL_FORMA_PAGO_ORIGINAL = "FORMA DE PAGO ORIGINAL"
COL_TIPO_PROGRAMA = "TIPO PROGRAMA"
COL_NACIONALIDAD = "NACIONALIDAD"
COL_PAIS_RESIDENCIA = "PAÍS DE RESIDENCIA"
COL_MANTIENE_MEDIO = "MANTIENE MEDIO PAGO"
COL_DIF_PI = "DIFERENCIA PI vs. IMPORTE PTE"
COL_DIF_FECHA_ORIGINAL = "DIFERENCIA FECHA 1ra CUOTA - ORIGINAL"
COL_DIF_FECHA_PROD = "DIFERENCIA FECHA PRODUCCIÓN - 1ra CUOTA"
COL_PCT_DTO = "% DTO"
COL_IMPORTE_COBRADO = "Importe Cobrado"

# Categóricas
CATEGORICAL_COLUMNS = [
    COL_DIA_CERO, COL_MEDIO_PAGO_PI, COL_MEDIO_PAGO_PTE, COL_FORMA_PAGO,
    COL_ASESOR, COL_FORMA_PAGO_ORIGINAL, COL_TIPO_PROGRAMA, COL_NACIONALIDAD,
    COL_PAIS_RESIDENCIA, COL_MANTIENE_MEDIO, COL_DIF_PI
]
BOOLEAN_OPTIONS = ["False", "True"]


#carga
@st.cache_resource(show_spinner=False)
def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        return joblib.load(f)


@st.cache_data(show_spinner=False)
def load_artifacts():
    df = pd.read_csv(TRAIN_DATA_PATH).drop(columns=DROP_COLUMNS, errors="ignore").dropna()

    category_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_COLUMNS if col in df.columns
    }

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    numeric_defaults = {
        col: int(round(df[col].median()))
        for col in feature_columns
        if col not in CATEGORICAL_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    }

    percent_defaults = {
        col: float(df[col].median()) for col in PERCENT_COLUMNS if col in df.columns
    }

    return category_options, feature_columns, numeric_defaults, percent_defaults


# Transformaciones
def transform_inputs(user_inputs: Dict[str, object], feature_columns: List[str]) -> pd.DataFrame:
    transformed = {}

    for col in feature_columns:
        if col not in user_inputs:
            raise KeyError(f"Falta el valor para: {col}")

        val = user_inputs[col]

        if col in DERIVED_BOOL_COLUMNS:
            transformed[col] = val == "True"
        elif col in PERCENT_COLUMNS:
            transformed[col] = float(val)
        elif isinstance(val, (int, float)):
            transformed[col] = val
        else:
            transformed[col] = str(val)

    return pd.DataFrame([transformed], columns=feature_columns)


# App
st.set_page_config(page_title="Clasificación de impago", page_icon=":moneybag:")

st.title("Clasificación de impago de un alumno")
st.write("Completa los campos y la aplicación predecirá si un alumno pagará su financiación.")

pipeline = load_pipeline()
category_options, feature_columns, numeric_defaults, percent_defaults = load_artifacts()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    dia_cero = col1.selectbox("Día cero", options=BOOLEAN_OPTIONS)
    facturacion = col1.number_input("Facturación neta (€)", min_value=0.0, value=float(numeric_defaults.get(COL_FACTURACION, 0)))
    pct_inscripcion = col1.slider("Porcentaje inscripción", 0.0, 1.0, float(percent_defaults.get(COL_PCT_INSCRIPCION, 0.15)))
    importe_inscripcion = facturacion * pct_inscripcion

    col1.number_input("Importe inscripción (€)", value=importe_inscripcion, disabled=True)

    importe_cobrado = col1.number_input(
        "Importe cobrado (€)", min_value=importe_inscripcion,
        value=max(float(numeric_defaults.get(COL_IMPORTE_COBRADO, 0)), importe_inscripcion)
    )
    importe_pendiente = max(facturacion - importe_cobrado, 0.0)
    col1.number_input("Importe pendiente (€)", value=importe_pendiente, disabled=True)

    cuotas = col1.number_input("Número de cuotas", min_value=0, value=numeric_defaults.get(COL_NUM_CUOTAS, 1))
    dif_fecha_orig = col1.number_input("Diferencia fecha 1ra cuota - original", min_value=-365, max_value=365, value=0)
    dif_fecha_prod = col1.number_input("Diferencia fecha producción - 1ra cuota", min_value=-400, max_value=1200, value=0)
    pct_dto = col1.slider("Porcentaje descuento", 0.0, 1.0, float(percent_defaults.get(COL_PCT_DTO, 0.0)))

    medio_pi = col2.selectbox("Medio de pago PI", options=category_options.get(COL_MEDIO_PAGO_PI, [""]))
    medio_pte = col2.selectbox("Medio de pago importe pendiente", options=category_options.get(COL_MEDIO_PAGO_PTE, [""]))
    forma_pago = col2.selectbox("Forma de pago", options=category_options.get(COL_FORMA_PAGO, [""]))
    asesor = col2.selectbox("Asesor", options=category_options.get(COL_ASESOR, [""]))
    forma_original = col2.selectbox("Forma de pago original", options=category_options.get(COL_FORMA_PAGO_ORIGINAL, [""]))
    tipo_programa = col2.selectbox("Tipo de programa", options=category_options.get(COL_TIPO_PROGRAMA, [""]))
    nacionalidad = col2.selectbox("Nacionalidad", options=category_options.get(COL_NACIONALIDAD, [""]))
    pais = col2.selectbox("País de residencia", options=category_options.get(COL_PAIS_RESIDENCIA, [""]))

    submitted = st.form_submit_button("Predecir")

if submitted:
    mantiene_medio = "True" if forma_pago == forma_original else "False"
    igual_medio_pago = "True" if medio_pi == medio_pte else "False"

    user_inputs = {
        COL_DIA_CERO: dia_cero,
        COL_FACTURACION: facturacion,
        COL_IMPORTE_INSCRIPCION: importe_inscripcion,
        COL_PCT_INSCRIPCION: pct_inscripcion,
        COL_MEDIO_PAGO_PI: medio_pi,
        COL_IMPORTE_PENDIENTE: importe_pendiente,
        COL_MEDIO_PAGO_PTE: medio_pte,
        COL_NUM_CUOTAS: cuotas,
        COL_FORMA_PAGO: forma_pago,
        COL_ASESOR: asesor,
        COL_FORMA_PAGO_ORIGINAL: forma_original,
        COL_TIPO_PROGRAMA: tipo_programa,
        COL_NACIONALIDAD: nacionalidad,
        COL_PAIS_RESIDENCIA: pais,
        COL_MANTIENE_MEDIO: mantiene_medio,
        COL_DIF_PI: igual_medio_pago,
        COL_DIF_FECHA_ORIGINAL: dif_fecha_orig,
        COL_DIF_FECHA_PROD: dif_fecha_prod,
        COL_PCT_DTO: pct_dto,
        COL_IMPORTE_COBRADO: importe_cobrado,
    }

    try:
        X = transform_inputs(user_inputs, feature_columns)
        prediction = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0][1]
    except Exception as e:
        st.error(f"Error al predecir: {e}")
    else:
        if importe_pendiente <= 0:
            prediction = 0
            proba = 0.0

        st.caption(f"Mantiene medio de pago: {'Sí' if mantiene_medio == 'True' else 'No'}")
        st.caption(f"Diferencia PI vs. Importe pendiente: {'Sí' if igual_medio_pago == 'True' else 'No'}")

        if prediction == 0:
            st.success(f"El alumno probablemente pagará. Probabilidad de impago: {proba:.1%}")
        else:
            st.error(f"El alumno probablemente **no pagará**. Probabilidad de impago: {proba:.1%}")
