import pandas as pd
import os

path_raw_data = os.path.join('data','raw')

def programa_to_id (programa, 
                    df_programas_id = 
                    pd.read_excel(os.path.join(
                        path_raw_data, 'programas-id.xlsx'
                    ))):
    
    id_selected = df_programas_id[df_programas_id['Programa']==programa]['ID PROGRAMA']
    return int(id_selected.values[0])
