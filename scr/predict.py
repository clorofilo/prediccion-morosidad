import pandas as pd
import pickle
import os

# Ruta al modelo desde la raíz del proyecto
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pipeline_final.pkl')

# Cargar el pipeline completo (preprocesamiento + modelo)
with open(model_path, "rb") as f:
    pipeline = pickle.load(f)

def preparar_input(usuario_dict):
    """
    Convierte un diccionario del formulario en un DataFrame.
    """
    input_df = pd.DataFrame([usuario_dict])
    return input_df

def predecir_morosidad(usuario_dict):
    """
    Recibe un diccionario con los datos del usuario y devuelve predicción y probabilidad.
    """
    df_input = preparar_input(usuario_dict)
    pred = pipeline.predict(df_input)[0]
    prob = pipeline.predict_proba(df_input)[0][1]  # probabilidad de ser moroso
    return pred, prob
