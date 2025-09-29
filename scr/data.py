import pandas as pd
import os 
from utils import (
    normalizar_pais,
    agrupar_pais
)

path_raw_data = os.path.join( 'data','raw')
df_deuda_or = pd.read_excel(os.path.join(path_raw_data,'estado_cartera.xlsx'))
df_seguimiento = pd.read_excel(os.path.join(path_raw_data,'Seguimiento cobros 2404.xlsx'))
df_convo_or= pd.read_excel(os.path.join(path_raw_data,'convo_ng_24.xlsx'), sheet_name='CONVOCATORIO')
df_paises_norm_or = pd.read_excel(os.path.join(path_raw_data,'paises.xlsx'), sheet_name="NORMALIZACION PAISES")
df_agrupaciones_paises = pd.read_excel(os.path.join(path_raw_data,'paises.xlsx'), sheet_name="PAISES")


#1. Tratar datos de convo
df_convo = df_convo_or.copy()
# Mantener 'ULT CONEXIÓN BB' para estudiar alumnos con riesgo de impago durante el curso
df_convo.drop(columns=['STATUS', 'SEGUIMIENTO', 'FECHA SEGUIMIENTO', 'PENDIENTE', 'FECHA FINALIZACION MATRICULA', 'FECHA INSCRIPCIÓN', 'SEMANA COMERCIAL PRODUCCION',
       'MES COMERCIAL PRODUCCION', 'SEMANA COMERCIAL INSCRIPCION',
       'MES COMERCIAL INSCRIPCION', 'OBSERVACIONES MATRICULACION','PROGRAMA','TIPO PROGRAMA', 'CAMPUS',
       'NOMBRES APELLIDOS',  'PROVINCIA',
       'ECTS MATRICULADOS', 'ECTS RECONOCIDOS', 'ECTS RECONOCIDOS 1er AÑO',
       'ECTS RECONOCIDOS 2do AÑO', 'ECTS RECONOCIDOS 3er AÑO',
       'ECTS NUEVOS 1er AÑO', 'ECTS CONVALIDADOS', 'IMPORTE DTO FECHA MATRICULACION', 'IMP. DTO. COMERCIAL',
       'IMP. DTO APERT/CIERRE', 'DESCRIPTIVO DTO. APERT/CIERRE',
       'IMP. DTO. REF.', 'DESCRIPTIVO REF.', 'IMP. DTO. OTROS',
       'DESCRIPTIVO DTO. OTROS', 'IMPORTE DTO PAGO CONTADO',
       'DESCRIPTIVO DTO. FINANCIERO', 'DTO. ETCS NO MATRICULADOS',
       'RECON. Y CONVAL. ETCS','PI PENDIENTE', 'FECHA CARTA COMPROMISO DE PAGO PI', 'FECHA CARTA COMPROMISO DE PAGO','EQUIPO', 'JEFE EQUIPO', 'DIRECTOR VENTAS', 'FECHA MODIFICACION',
       'SEMANA COMERCIAL MODIFICACION', 'MES COMERCIAL MODIFICACION',
       'CONVOCATORIA ORIGEN', 'PROGRAMA ORIGEN', 'OBSERVACIONES CAMBIO CONV.',
       'TASA', 'FACTURACION NETA ORIGINAL', 'NUMERO DE CUOTAS ORIGINAL','FECHA REC MAIL', 'FECHA BAJA',
       'SEMANA COMERCIAL BAJA', 'MES COMERCIAL BAJA', 'FECHA GI',
       'PERIODO DESISTIMIENTO', 'MOTIVO DE BAJA', 'OBSERVACIONES BAJA',
       'CORRESPONDE DEVOLUCION', 'MES COMISIONES', 'COMISION CONTADO',
       'MES PAGO COMISION CONTADO', 'OBSERVACIONES COMISION','ULT CONEXIÓN BB', 'SITUACION COBROS NACS',
       'VERIFICACION SITUACION COBRO',
       'IMPORTE PENDIENTE PAGO (CIERRE CONTABLE)', 'PROPUESTA CIERRE CONTABLE',
       'OBSERVACIONES CIERRE CONTABLE', 'INCIDENCIAS SIN RESOLVER',
       'LINEA COMPLETA', 'Forma de Pago en Atenea'], inplace=True)

#Crear columna "TIPO PROGRAMA" segun MST o MBA.
df_convo['ID PROGRAMA'] = df_convo['ID PROGRAMA'].astype(str).str[:-2]
df_convo['TIPO PROGRAMA'] = df_convo['ID PROGRAMA'].astype(str).str[-3:].apply(lambda x: 'MBA' if x in ['990', '900', '902', '620'] else 'MST')

#Rellenar columna Forma de pago original
df_convo['FORMA DE PAGO ORIGINAL'].fillna(df_convo['FORMA DE PAGO'], inplace=True)

#Rellenar columna Medio de pago importe pendiente
df_convo['MEDIO PAGO IMPORTE PENDIENTE'].fillna('SIN IMPORTE PTE', inplace=True)


# Convertir las variables pertientes a numericas
columnas_numericas = ['PRECIO CURSO', 'FACTURACIÓN NETA', 'IMPORTE INSCRIPCIÓN', '% INSCRIPCION',
       'IMPORTE PENDIENTE PAGO']
for col in columnas_numericas:
    df_convo[col] = pd.to_numeric(df_convo[col], errors='coerce')

df_convo['NUMERO DE CUOTAS'] = df_convo['NUMERO DE CUOTAS'].astype('Int64')

df_convo = df_convo.dropna(subset=['ID NACS'])
df_convo['ID NACS'] = df_convo['ID NACS'].astype(int)

# Convertir las variables pertientes a fechas
columnas_fechas = ['FECHA PRODUCCIÓN',  'FECHA 1ra CUOTA',
       'FECHA 1ra CUOTA ORIGINAL',]
for col in columnas_fechas:
    df_convo[col] = pd.to_datetime(df_convo[col], errors='coerce')

# Convertir las variables pertientes a booleanas
df_convo['DIA CERO'] = df_convo['DIA CERO'].map(lambda x: True if x == 'SI' else False)

#Normalizar paises
df_paises_norm = df_paises_norm_or.copy()
df_paises_norm.drop_duplicates(subset='PAIS A NORMALIZAR', keep='first', inplace=True)
df_paises_norm.drop(columns=['FUENTE-PAIS', 'FUENTE'], inplace=True)
df_convo = normalizar_pais(df_convo, ['NACIONALIDAD', 'PAÍS DE RESIDENCIA'], df_paises_norm)

#Agrupar Paises
df_agrupaciones_paises.drop_duplicates(subset='PAIS', keep='first', inplace=True)
df_agrupaciones_paises = df_agrupaciones_paises[['PAIS', 'CLASIFICACION_MKT']]
df_convo = agrupar_pais(df_convo, ['NACIONALIDAD', 'PAÍS DE RESIDENCIA'], df_agrupaciones_paises)

# Convertir las variables pertientes a categoricas
columnas_categoricas = ['TIPOLOGIA MATRICULA', 'TIPOLOGIA ALUMNO', 'ID OPORTUNIDAD',
       'DIA CERO', 'CONVOCATORIA', 'ID PROGRAMA',
       'TIPO PROGRAMA', 'NACIONALIDAD', 'PAÍS DE RESIDENCIA', 
       'MEDIO PAGO PI', 'MEDIO PAGO IMPORTE PENDIENTE', 
       'FORMA DE PAGO', 'ASESOR', 
       'FORMA DE PAGO ORIGINAL', 'AGRUPACION NACIONALIDAD', 'AGRUPACION PAÍS DE RESIDENCIA'
         ]
for col in columnas_categoricas:
    df_convo[col] = df_convo[col].astype('category')

#CREACION DE NUEVAS COLUMNAS
#Medio de pago PI = resto de pagos
df_convo['MANTIENE MEDIO PAGO'] = df_convo['MEDIO PAGO PI'].astype(str) == df_convo['MEDIO PAGO IMPORTE PENDIENTE'].astype(str)

#Cambio en la fecha 1a cuota
df_convo['DIFERENCIA FECHA 1ra CUOTA - ORIGINAL'] = (df_convo['FECHA 1ra CUOTA'] - df_convo['FECHA 1ra CUOTA ORIGINAL']).dt.days
df_convo.fillna({'DIFERENCIA FECHA 1ra CUOTA - ORIGINAL': 0}, inplace=True)

#Diferencia entre la fecha de produccion y la fecha de 1ra cuota
df_convo['DIFERENCIA FECHA PRODUCCIÓN - 1ra CUOTA'] = (df_convo['FECHA 1ra CUOTA'] - df_convo['FECHA PRODUCCIÓN']).dt.days
df_convo.fillna({'DIFERENCIA FECHA PRODUCCIÓN - 1ra CUOTA': 0}, inplace=True)

#% de Descuento
df_convo ['% DTO'] = 1-(df_convo['FACTURACIÓN NETA']/df_convo['PRECIO CURSO'])

#Tratar datos de estado cartera
df_deuda  =df_deuda_or.copy()
#Crear FP1 y Cuotas
df_deuda['FP1'] = df_deuda['Forma Pago'].str[0]
df_deuda['Cuotas'] = df_deuda['Forma Pago'].str[-2:]

#Convertir los NAs
df_deuda.fillna({
    'Importe Impagado Actual': 0,
    'Importe Neto Factura': 0,
    'Importe Cartera': 0,
    'Importe Vdo': 0,
    'Importe Recobrado': 0,
    'Importe Cobrado': 0,
    'Importe No Vencido': 0,
    'Marca': '',
    'Convocatoria': '',
    'FP1': '',
    'Cuotas': 0
}, inplace=True)

#Agrupar por 'Cod. NACS' y sumar las columnas pertinentes
df_deuda = df_deuda.groupby('Cod. NACS').aggregate({
    'Marca': 'first',
    'Convocatoria': 'first',
    'FP1': 'first',
    'Cuotas': 'first', 
    'Importe Impagado Actual': 'sum',
    'Importe Neto Factura': 'sum',
    'Importe Cartera': 'sum',
    'Importe Vdo': 'sum',
    'Importe Recobrado': 'sum',
    'Importe Cobrado': 'sum',
    'Importe No Vencido': 'sum'
    })

#Nueva columna para los morosos
df_deuda['Moroso'] = df_deuda['Importe Impagado Actual'] > 0

#Elimina inicio de NACS para para poder hacer el Join
df_deuda.index = df_deuda.index.astype(str).str.replace(r'^(44|40)', '', regex=True)
df_deuda.index = df_deuda.index.astype(int)

#Recalcular % de Impago actual Vdo
df_deuda['% Impagado Actual Vdo'] = (df_deuda['Importe Impagado Actual']/df_deuda['Importe Vdo'])
#Selecciona las columnas de df_convo que se quieren unir con df_deuda
columnas_deuda = ['Moroso', '% Impagado Actual Vdo', 'Importe Impagado Actual']

#Unir df_convo con df_deuda para saber si es moroso o no
df_convo_deuda = pd.merge(df_convo, df_deuda[columnas_deuda], how='left', left_on='ID NACS', right_index=True)
#Eliminar filas sin data
df_convo_deuda = df_convo_deuda.dropna(subset=['TIPOLOGIA MATRICULA', 'Moroso'])

df_convo_deuda.drop(columns= ['TIPOLOGIA MATRICULA', 'TIPOLOGIA ALUMNO', 'ID OPORTUNIDAD', 'ID NACS',  'CONVOCATORIA'], inplace = True)

df_convo_deuda.to_csv(os.path.join('data','processed','data.csv'), index=False)