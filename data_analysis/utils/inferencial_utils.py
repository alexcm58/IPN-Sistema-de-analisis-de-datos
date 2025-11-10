import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import uuid
from django.conf import settings

from scipy import stats

# Este script contiene funciones para realizar análisis inferencial de datos, incluyendo regresión lineal, regresión logística, intervalo de confianza y prueba t.
# Se obtienen resultados y se envían a la vista para su visualización.
# Incluye las funciones regresion_linear, regresion_logistica, calcular_intervalo_confianza y prueba_t.


def regresion_linear(df, variable_dependiente, variables_independientes):
    resultado = {
        'Errores': []
    }

    try:
        # Preparar datos
        X = df[variables_independientes]
        y = df[variable_dependiente]

        # Obtener las primeras 10 filas para las columnas especificadas
        columnas_unicas = list(set(variables_independientes + [variable_dependiente]))
        primeras_filas = df[columnas_unicas].head(10)

        # Dividir los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inicializar y entrenar el modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Realizar predicciones con el conjunto de prueba
        y_pred = modelo.predict(X_test)

        # Calcular métricas de rendimiento
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Crear una lista para los gráficos
        graficos = []

        # Generar un gráfico para cada variable independiente
        for i, var_indep in enumerate(variables_independientes):
            # Seleccionar la columna de datos correspondiente
            X_var = X_test[var_indep].values.reshape(-1, 1)

            # Ajustar el modelo para una sola variable independiente
            modelo_var = LinearRegression()
            modelo_var.fit(X_var, y_test)
            y_pred_var = modelo_var.predict(X_var)

            # Ordenar los puntos para el gráfico de línea
            sorted_axis = np.argsort(X_var, axis=0).flatten()
            X_var_sorted = X_var[sorted_axis]
            y_pred_var_sorted = y_pred_var[sorted_axis]

            # Preparar puntos de la línea de ajuste
            puntos_linea_ajuste = [{"x": float(X_var_sorted[i]), "y": float(y_pred_var_sorted[i])} for i in range(len(X_var_sorted))]

            # Preparar puntos observados
            puntos_observados = [{"x": float(X_var[i][0]), "y": float(y_test.iloc[i])} for i in range(len(X_var))]
            
            # Agregar al gráfico
            graficos.append({
                "tipo_grafico": "scatter_linea",
                "titulo": f"Relación lineal entre {var_indep} y {variable_dependiente}",
                "datos": {
                    "puntos_observados": puntos_observados,
                    "puntos_linea_ajuste": puntos_linea_ajuste,
                }
            })
        
        # Preparar el resultado para devolver
        reporte_data = {
            "titulo": "Reporte - Análisis de Regresión Lineal",
            "descripcion": f"Análisis de regresión lineal de la variable dependiente (X): [{variable_dependiente}] con las variables independientes (Y): [{', '.join(variables_independientes)}].",
            "tipo_analisis": "regresion_lineal",
            "resultados": {
                "estadisticas": {
                    "MSE": mse,
                    "R²": r2
                },
                "coeficientes": dict(zip(variables_independientes, modelo.coef_)),
                "intercepto": modelo.intercept_,
                "nombres_columnas": columnas_unicas,
                "primeras_filas": primeras_filas.to_dict('records'), 
                "graficos": graficos
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None

    return df, reporte_data, resultado


def regresion_logistica(df, variable_dependiente, variables_independientes, umbral=0.5):
    resultado = {
        'Errores': []
    }
    try:
        # Preparar datos
        X = df[variables_independientes]
        y = label_binarize(df[variable_dependiente].values, classes=[0, 1])[:, 0]

        # Obtener las primeras 10 filas para las columnas especificadas
        columnas_unicas = list(set(variables_independientes + [variable_dependiente]))
        primeras_filas = df[columnas_unicas].head(10)
       
        # Dividir los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inicializar y entrenar el modelo de regresión logística
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_train, y_train)

        # Realizar predicciones con el conjunto de prueba
        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= umbral).astype(int)

        # Calcular métricas de rendimiento
        matriz_confusion = confusion_matrix(y_test, y_pred).tolist()  # Convertir a lista para visualización

        
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Preparar datos para el gráfico de dispersión con línea de ajuste
        dispersion_data = None
        if len(variables_independientes) == 1:
            variable_indep = X_test[variables_independientes[0]]
            # Ordenar los valores para la línea
            sorted_indices = np.argsort(variable_indep)
            variable_indep_sorted = variable_indep.iloc[sorted_indices]
            y_pred_proba_sorted = y_pred_proba[sorted_indices]

            dispersion_data = {
                "x": variable_indep_sorted.tolist(),
                "y": y_test[sorted_indices].tolist(),
                "y_model": y_pred_proba_sorted.tolist()
            }

        # Generar el reporte de datos
        reporte_data = {
            "titulo": "Reporte - Análisis de Regresión Logística",
            "descripcion": f"Análisis de regresión logística para la variable dependiente (X): [{variable_dependiente}] con las variables independientes (Y): [{', '.join(variables_independientes)}] y un umbral de {umbral}.",
            "tipo_analisis": "regresion_logistica",
            "resultados": {
                "matriz_confusion": matriz_confusion,
                "accuracy": acc,
                "recall": recall,
                "precision": precision,
                "f1_score": f1,
                "auc": roc_auc,
                "coeficientes": dict(zip(variables_independientes, modelo.coef_.flatten())),
                "intercepto": modelo.intercept_[0],
                "nombres_columnas": columnas_unicas, 
                "primeras_filas": primeras_filas.to_dict('records'), 
                "graficos": [
                    {
                        "tipo_grafico": "curva_roc",
                        "titulo": "Gráfico de curva ROC",
                        "datos": {
                            "puntos_curva_roc": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in range(len(fpr))],
                            "roc_auc": roc_auc
                        }
                    }
                ]
            }
        }

        if dispersion_data:
            reporte_data["resultados"]["graficos"].append({
                "tipo_grafico": "dispersion",
                "titulo": f"Gráfico de dispersión con línea de ajuste para {variables_independientes[0]}",
                "datos": dispersion_data
            })
    except Exception as e: 
        resultado['Errores'].append(str(e))
        reporte_data = None
    
    return df, reporte_data, resultado



# Colores del IPN
color_ipn_1 = "#800020"  # Color vino
color_ipn_2 = "#FFD700"  # Color dorado
color_ipn_3 = "#FFFFFF"  # Color blanco

def generar_grafico_intervalo_confianza(columna, media, desviacion_estandar, n, intervalo_inferior, intervalo_superior, alpha=0.05):
    # Generar datos para la curva de distribución
    x = np.linspace(media - 4*desviacion_estandar, media + 4*desviacion_estandar, 1000)
    y = stats.norm.pdf(x, media, desviacion_estandar / np.sqrt(n))
    
    plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
    plt.plot(x, y, color='black')
    
    # Rellenar el intervalo de confianza con el color del IPN
    plt.fill_between(x, y, where=(x >= intervalo_inferior) & (x <= intervalo_superior), color=color_ipn_2, alpha=0.5)
    
    # Añadir líneas y anotaciones para la media y los límites
    plt.axvline(media, color=color_ipn_1, linestyle='-', label=f'la media = {media:.2f}')
    plt.axvline(intervalo_inferior, color=color_ipn_1, linestyle='--', label=f'límite inferior = {intervalo_inferior:.2f}')
    plt.axvline(intervalo_superior, color=color_ipn_1, linestyle='--', label=f'límite superior = {intervalo_superior:.2f}')
    
    # Anotaciones en el gráfico
    plt.text(media, max(y) * 0.8, f'Media = {media:.2f}', horizontalalignment='center', fontsize=12, color=color_ipn_1)
    plt.text(intervalo_inferior, max(y) * 0.6, f'Límite inferior = {intervalo_inferior:.2f}', horizontalalignment='center', fontsize=12, color=color_ipn_1)
    plt.text(intervalo_superior, max(y) * 0.6, f'Límite superior = {intervalo_superior:.2f}', horizontalalignment='center', fontsize=12, color=color_ipn_1)
    
    # Anotar el valor del intervalo de confianza
    intervalo_confianza = intervalo_superior - intervalo_inferior
    plt.text(media, max(y) * 0.4, f'Intervalo de Confianza = {intervalo_confianza:.2f}', horizontalalignment='center', fontsize=12, color=color_ipn_1)
    
    # Ajustes del gráfico
    plt.title(f'Nivel de Confianza del 95% para {columna}', fontsize=16, color=color_ipn_1)
    plt.xlabel('Valores', fontsize=14, color=color_ipn_1)
    plt.ylabel('Densidad de probabilidad', fontsize=14, color=color_ipn_1)
    plt.legend(loc='upper right', fontsize=12)
    
    # Cambiar el color del fondo
    plt.gca().set_facecolor(color_ipn_3)
    plt.gcf().set_facecolor(color_ipn_3)
    
    # Ajustar los límites del eje x para mejorar la visualización
    margen = (intervalo_superior - intervalo_inferior) * 0.1
    plt.xlim(intervalo_inferior - margen, intervalo_superior + margen)
    
    unique_filename = str(uuid.uuid4()) + '.png'
    file_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename)
    
    plt.savefig(file_path)
    plt.close()
    
    return unique_filename

def calcular_intervalo_confianza(df, columnas, alpha=0.05):
    resultado = {'Errores': []}
    reporte_data = None

    try:
        # Calcular el intervalo de confianza para cada columna
        resultados_intervalo_confianza = {}
        for columna in columnas:
            datos_columna = df[columna]
            if datos_columna.empty:  # Verificar si la serie está vacía
                resultado['Errores'].append(f"No se especificaron datos para la columna {columna}.")
            else:
                media = datos_columna.mean()
                desviacion_estandar = datos_columna.std()
                n = len(datos_columna)
                error_estandar = desviacion_estandar / np.sqrt(n)
                margen_error = stats.norm.ppf(1 - alpha / 2) * error_estandar
                intervalo_inferior = media - margen_error
                intervalo_superior = media + margen_error

                errores_inferiores = media - intervalo_inferior
                errores_superiores = intervalo_superior - media

                unique_filename = generar_grafico_intervalo_confianza(columna, media, desviacion_estandar, n, intervalo_inferior, intervalo_superior, alpha)
                
                resultados_intervalo_confianza[columna] = {
                    'Media': media.item() if isinstance(media, (np.integer, np.floating)) else media,
                    'Error_Estandar': error_estandar.item() if isinstance(error_estandar, (np.integer, np.floating)) else error_estandar,
                    'Intervalo_Inferior': intervalo_inferior.item() if isinstance(intervalo_inferior, (np.integer, np.floating)) else intervalo_inferior,
                    'Intervalo_Superior': intervalo_superior.item() if isinstance(intervalo_superior, (np.integer, np.floating)) else intervalo_superior,
                    'Error_Inferior': errores_inferiores.item() if isinstance(errores_inferiores, (np.integer, np.floating)) else errores_inferiores,
                    'Error_Superior': errores_superiores.item() if isinstance(errores_superiores, (np.integer, np.floating)) else errores_superiores,
                    'Intervalo_Confianza': intervalo_superior - intervalo_inferior,
                    'grafico_path': unique_filename
                }

        # Convertir primeras filas a tipos nativos de Python
        primeras_filas = df[columnas].head(10).to_dict('records')
        primeras_filas = [{k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in fila.items()} for fila in primeras_filas]

        # Crear datos del reporte
        reporte_data = {
            "titulo": "Reporte - Análisis de Intervalo de Confianza",
            "descripcion": f"Análisis del intervalo de confianza realizado sobre las columnas: [{', '.join(columnas)}].",
            "tipo_analisis": "intervalo_confianza",
            "resultados": {
                "nombres_columnas": columnas,
                "primeras_filas": primeras_filas,
                "intervalo_confianza": resultados_intervalo_confianza
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
    
    return df, reporte_data, resultado


def generar_grafico_prueba_t(columna, media, desviacion_estandar, t, p_valor, df, tipo_prueba, nivel_significancia):
    # Generar valores de la distribución normal para la gráfica
    valores_normales = np.linspace(media - 4*desviacion_estandar, media + 4*desviacion_estandar, 1000)
    prob_densidad = (1 / (desviacion_estandar * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((valores_normales - media) / desviacion_estandar)**2)

    # Gráfico de la distribución de los datos
    plt.figure(figsize=(10, 5))
    plt.plot(valores_normales, prob_densidad, label='Distribución Normal')  

    # Añadir las líneas de la zona de rechazo
    z_critical = np.abs(stats.norm.ppf(nivel_significancia / 2)) if tipo_prueba == 'dos_colas' else np.abs(stats.norm.ppf(nivel_significancia))
    lower_bound = media - z_critical * desviacion_estandar
    upper_bound = media + z_critical * desviacion_estandar

    if tipo_prueba == 'dos_colas':
        plt.axvline(lower_bound, color='r', linestyle='dashed', linewidth=1.5)
        plt.axvline(upper_bound, color='r', linestyle='dashed', linewidth=1.5)
        plt.fill_betweenx(prob_densidad, valores_normales, lower_bound, where=(valores_normales <= lower_bound), color='red', alpha=0.5)
        plt.fill_betweenx(prob_densidad, valores_normales, upper_bound, where=(valores_normales >= upper_bound), color='red', alpha=0.5)
    elif tipo_prueba == 'una_cola_der':
        plt.axvline(upper_bound, color='r', linestyle='dashed', linewidth=1.5)
        plt.fill_betweenx(prob_densidad, valores_normales, upper_bound, where=(valores_normales >= upper_bound), color='red', alpha=0.5)
    elif tipo_prueba == 'una_cola_izq':
        plt.axvline(lower_bound, color='r', linestyle='dashed', linewidth=1.5)
        plt.fill_betweenx(prob_densidad, valores_normales, lower_bound, where=(valores_normales <= lower_bound), color='red', alpha=0.5)

    # Decoración del gráfico
    plt.title(f'Gráfico de la Prueba T para {columna} (t={t:.2f}, p={p_valor:.3f})')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()

    # Guardar la imagen
    unique_filename = str(uuid.uuid4()) + '.png'
    graph_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename)
    plt.savefig(graph_path, dpi=400)
    plt.close()

    return unique_filename

    
def prueba_t(df, columnas, mu=0, tipo_prueba='dos_colas', nivel_significancia=0.05):
    resultado = {'Errores': []}
    reporte_data = None

    try:
        resultados_prueba_t = {}
        columnas_iguales = []

        for columna in columnas:
            datos_columna = df[columna].dropna()  # Eliminar valores nulos
            if datos_columna.empty:
                resultado['Errores'].append(f"No se especificaron datos para la columna {columna}.")
            elif len(datos_columna) < 2:
                resultado['Errores'].append(f"No hay suficientes datos en la columna {columna} para realizar la prueba t.")
            elif datos_columna.std() == 0:
                columnas_iguales.append(columna)
            else:
                media = datos_columna.mean()
                desviacion_estandar = datos_columna.std()
                n = len(datos_columna)
                
                # Calculo de p-valor adaptado al tipo de prueba
                if tipo_prueba == 'dos_colas':
                    estadistico_t, p_valor = stats.ttest_1samp(datos_columna, mu)
                elif tipo_prueba == 'una_cola_izq':
                    estadistico_t, p_valor_dos_colas = stats.ttest_1samp(datos_columna, mu)
                    p_valor = p_valor_dos_colas / 2 if estadistico_t < 0 else 1
                elif tipo_prueba == 'una_cola_der':
                    estadistico_t, p_valor_dos_colas = stats.ttest_1samp(datos_columna, mu)
                    p_valor = p_valor_dos_colas / 2 if estadistico_t > 0 else 1

                df_columna = n - 1  # Grados de libertad
                grafico_path = generar_grafico_prueba_t(columna, media, desviacion_estandar, estadistico_t, p_valor, df_columna, tipo_prueba, nivel_significancia)

                conclusion = "Rechazar H0" if p_valor < nivel_significancia else "No rechazar H0"
                resultados_prueba_t[columna] = {
                    'Media': media,
                    'Desviacion_Estandar': desviacion_estandar,
                    'T': estadistico_t,
                    'P_valor': p_valor,
                    'Conclusion': conclusion,
                    'grafico_path': grafico_path
                }

        if columnas_iguales:
            resultado['Errores'].append(f"Todos los valores en las siguientes columnas son iguales: {', '.join(columnas_iguales)}. Por favor, elija columnas con valores variables para el análisis.")

        primeras_filas = df[columnas].head(10).to_dict('records')
        primeras_filas = [{k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in fila.items()} for fila in primeras_filas]

        reporte_data = {
            "titulo": "Reporte - Prueba de Hipotesis",
            "descripcion": f"Prueba de hipotesis realizada sobre las columnas: [{', '.join(columnas)}], Mu: {mu}, Tipo de Prueba: {tipo_prueba}, Nivel de Significancia: {nivel_significancia}.",
            "tipo_analisis": "prueba_t_una_muestra",
            "resultados": {
                "nombres_columnas": columnas,
                "primeras_filas": primeras_filas,
                "prueba_t": resultados_prueba_t
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None

    return df, reporte_data, resultado