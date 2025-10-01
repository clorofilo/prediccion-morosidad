import streamlit as st
import numpy as np
import joblib
from streamlit_drawable_canvas import st_canvas
from scr.feature import programa_to_id

#Cargar modelo
rf_clf = joblib.load('models/modelo_random_forest_optimizado.pkl')

#Cargar conversor categoricas
encoder = joblib.load('scr/pipeline_cat_colums.pkl')

#cargar tabla
