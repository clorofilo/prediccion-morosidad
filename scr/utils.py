import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr, levene, kruskal, mannwhitneyu, chi2_contingency, ttest_ind



#Normalizar paises
def normalizar_pais(df, columnas_a_normalizar, df_paises_norm):
    for col in columnas_a_normalizar:
        df = pd.merge(
            df,
            df_paises_norm,
            how='left',
            left_on=col,
            right_on='PAIS A NORMALIZAR'
        )
        df.drop(columns=['PAIS A NORMALIZAR', col], inplace=True)
        df.rename(columns={'PAIS': col}, inplace=True)
    return df
#Agrupacion de paises
def agrupar_pais(df, columnas_paises, df_agrupaciones_paises):
    for col in columnas_paises:
        df = pd.merge(
            df,
            df_agrupaciones_paises,
            how='left',
            left_on=col,
            right_on='PAIS'
        )
        df.drop(columns=['PAIS'], inplace=True)
        df.rename(columns={'CLASIFICACION_MKT': f'AGRUPACION {col}'}, inplace=True)
    return df

#comprobar normalidad de las variables
def comprobar_normalidad(df, var1, var2):
    _, p1 = shapiro(df[var1])
    _, p2 = shapiro(df[var2])
    
    if p1 > 0.05 and p2 > 0.05:
        print("Ambas variables tienen distribución normal → puedes usar Pearson.")
        return True
    else:
        print("Alguna variable no es normal → mejor usar Spearman.")
        return False
    # Spearman
    spearman_corr, p_spearman = spearmanr(df[var1], df[var2])
    print(f"Spearman: ρ = {spearman_corr:.3f}, p = {p_spearman:.3f}")

def correlacion_variables(df, var1, var2):
    print (f"Comprobando correlación entre {var1} y {var2} \n")
    normalidad = comprobar_normalidad(df, var1, var2)
    
    if normalidad:
        corr, p = spearmanr(df[var1], df[var2])
        print(f"Normalidad en la distribución => Correlación de Pearson: r = {corr:.3f}, p = {p:.3f}")
        if p < 0.05:
            print("La correlación es estadísticamente significativa.")
        else:
            print("La correlación no es estadísticamente significativa.")
    else:
        corr, p = spearmanr(df[var1], df[var2])
        print(f"Sin normalidad en la distribución => Correlación de Spearman: ρ = {corr:.3f}, p = {p:.3f}")
        if p < 0.05:
            print("La correlación es estadísticamente significativa.")
        else:
            print("La correlación no es estadísticamente significativa.")

# Comprobar normalidad y varianza de los grupos
def comprobar_nomalidad_y_varianza(grupos): 
    # 1. Shapiro-Wilk test para normalidad
    for name, values in grupos:
        stat, p = shapiro(values)
        print(f'{name}: p normalidad = {p:.3f}')
        if p < 0.05:
            print(f"{name} no sigue una distribución normal.")
        else:
            print(f"{name} sigue una distribución normal.")

    # 2. Levene para varianzas iguales
    stat, p = levene(*[g for name, g in grupos])
    print(f'Levene p = {p:.3f}')
    if p < 0.05:
        print("Las varianzas son significativamente diferentes.")
    else:
        print("Las varianzas son homogéneas.")

#Calculo de la fuerza de asociacion entre dos variables sin normalidad y / o con varianzas diferentes entre dos grupos
def calcular_fuerza_asociacion_2_grupos(grupos):
    stat, p = mannwhitneyu(*[g for name, g in grupos])
    print(f'Mann-Whitney U: U = {stat:.2f}, p = {p:.3f}')
    if p < 0.05:
        print("Hay diferencias significativas entre los grupos.")
    else:
        print("No hay diferencias significativas entre los grupos.")

#Calculo de la fuerza de asociacion entre dos variables sin normalidad y / o con varianzas diferentes entre más de dos grupos
def calcular_fuerza_asociacion_mas_2_grupos(grupos):
    stat, p = kruskal(*[g for name, g in grupos])
    print(f'Kruskal-Wallis: H = {stat:.2f}, p = {p:.3f}')
    if p < 0.05:
        print("Hay diferencias significativas entre los grupos.")
    else:
        print("No hay diferencias significativas entre los grupos.")


def cramers_v(tabla):
    chi2 = chi2_contingency(tabla)[0]
    n = tabla.sum().sum()
    return np.sqrt(chi2 / n)



def calcular_independencia_categoricas(df, var1, var2):
    tabla = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(tabla)
    V = cramers_v(tabla)
    return var1, var2, chi2, p, V

# Crear nueva columna de agrupación personalizada para agrupar el tramo de deuda
def agrupar_tramo(tramo):
    if tramo < 14:
        return '0-13%'
    elif 14 <= tramo < 16:
        return '14-15%'
    elif 16 <= tramo <= 20:
        return '16-20%'
    else:
        return 'Resto'
    
def cohens_d(grupo1, grupo2):
    n1, n2 = len(grupo1), len(grupo2)
    s1, s2 = grupo1.std(ddof=1), grupo2.std(ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    d = (grupo1.mean() - grupo2.mean()) / s_pooled
    return d

def r_mannwhitney(u_stat, n1, n2):
    z = (u_stat - (n1 * n2) / 2) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    r = abs(z) / np.sqrt(n1 + n2)
    return r

def prueba_comparacion_2grupos_variable(df_num_col, variable_grupo, alpha=0.05):
    df_temp = df_num_col.to_frame(name='valor')
    df_temp['grupo'] = variable_grupo

    grupos_unicos = df_temp['grupo'].dropna().unique()
    if len(grupos_unicos) != 2:
        print(f"⚠️ La variable de grupo no tiene exactamente 2 categorías: {grupos_unicos}")
        return

    grupos = [(str(g), df_temp[df_temp['grupo'] == g]['valor'].dropna()) for g in grupos_unicos]

    print(f"\n📊 Variable: {df_num_col.name}")
    normalidad = {}
    
    print("▶ PRUEBA DE NORMALIDAD (Shapiro-Wilk):")
    for name, values in grupos:
        stat, p = shapiro(values)
        normal = p >= alpha
        normalidad[name] = normal
        print(f"  - Grupo {name}: p = {p:.3f} → {'Normal' if normal else 'No normal'}")

    print("▶ PRUEBA DE IGUALDAD DE VARIANZAS (Levene):")
    stat, p_var = levene(*[g for _, g in grupos])
    iguales_varianzas = p_var >= alpha
    print(f"  - p = {p_var:.3f} → {'Homogéneas' if iguales_varianzas else 'Diferentes'} varianzas")

    todos_normales = all(normalidad.values())

    print("▶ PRUEBA DE DIFERENCIAS ENTRE GRUPOS:")
    grupo1 = grupos[0][1]
    grupo2 = grupos[1][1]

    if todos_normales:
        t_stat, p_test = ttest_ind(grupo1, grupo2, equal_var=iguales_varianzas)
        print(f"  - t-test {'(igual varianza)' if iguales_varianzas else '(Welch)'} → p = {p_test:.3f}")
        metodo = "t-test"
        stat = t_stat
    else:
        u_stat, p_test = mannwhitneyu(grupo1, grupo2, alternative='two-sided')
        print(f"  - Mann-Whitney U → p = {p_test:.3f}")
        metodo = "Mann-Whitney"
        stat = u_stat

    # Tamaño del efecto
    if p_test < alpha:
        if metodo == "t-test":
            effect_size = cohens_d(grupo1, grupo2)
            print(f"Tamaño del efecto (Cohen's d): {effect_size:.3f}")
        else:
            effect_size = r_mannwhitney(stat, len(grupo1), len(grupo2))
            print(f"Tamaño del efecto (r de Mann-Whitney): {effect_size:.3f}")
    else:
        effect_size = None
        print(f"No se encontraron diferencias significativas entre grupos según {metodo}.")

    return p, p_var, metodo, stat, p_test, effect_size