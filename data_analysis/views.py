import json
from django.shortcuts import render
from http.client import HTTPResponse
from django.shortcuts import redirect
import pandas as pd
from .utils.data_analysis_utils import guardar_resultado_en_sesion, preparar_datos_para_get
from utils.csv_utils import gestionar_version_archivo, leer_csv_o_error, guardar_csv
from utils.tablas_utils import crear_inicio_tabla
import numpy as np
import re 
from .utils import inferencial_utils
from .utils import correlacion_utils
from .utils import machine_learning_utils
from .utils import descriptiva_utils

from django.shortcuts import redirect

# Vistas de análisis de datos que son la conexión entre las plantillas y las funciones de utilidad.
# Aquí están todas las vistas dedicadas al analisis de datos.
# Se incluyen las funciones: distribucion_frecuencias, tendencia_central, variabilidad, regresion_lineal, regresion_logistica, correlacion_pearson, correlacion_spearman, kmeans_clustering, arbol_decision_clasificacion, intervalo y prueba_hipotesis.

def distribucion_frecuencias(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')

    # Obtener nombres de columnas categóricas
    columnas_categoricas = df.select_dtypes(include=['category']).columns.tolist()

    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['category'])

    # Variables para el mensaje de error
    error_message = None

    # Obtener las primeras 10 filas para las columnas seleccionadas
    columnas_a_manipular = request.session.get('columnas_a_manipular', [])
    primeras_filas = df[columnas_a_manipular].head(10).to_dict(orient='list') if columnas_a_manipular else {}

    # Manejo de POST
    if request.method == 'POST':
        columnas_a_manipular = request.POST.getlist('columnas_a_manipular')

        if not columnas_a_manipular:
            error_message = "No se especificaron columnas para el análisis."
        else:
            # Calcular distribución de frecuencias utilizando la función en utils.py
            df, reporte_data, resultado = descriptiva_utils.calcular_distribucion_frecuencias(df, columnas_a_manipular)
            if resultado['Errores']:
                error_message = ', '.join(resultado['Errores'])
            else:
                # Guardar los resultados en la sesión para su posterior uso
                request.session['reporte_data'] = json.dumps(reporte_data)

                # Redirigir a la página del informe si no hay errores
                return redirect('report:reporte_distribucion_frecuencias', file_name=file_name)

    # Pasar las columnas al contexto del template incluso si hay un mensaje de error
    return render(request, 'data_analysis/distribucion_frecuencias.html', {
        'file_name': file_name,
        'columnas': columnas,
        'columnas_categoricas': columnas_categoricas,
        'dataframe': df_html,
        'error_message': error_message,
        "columna_seleccionada": columnas_a_manipular
    })

def tendencia_central(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')

    # Obtener nombres de columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])

    # Variables para el mensaje de error
    error_message = None

    # Obtener las primeras 10 filas para las columnas seleccionadas
    columnas_a_manipular = request.session.get('columnas_a_manipular', [])
    primeras_filas = df[columnas_a_manipular].head(10).to_dict(orient='list') if columnas_a_manipular else {}

    # Manejo de POST
    if request.method == 'POST':
        columnas_a_manipular = request.POST.getlist('columnas_a_manipular')

        if not columnas_a_manipular:
            error_message = "No se especificaron columnas para el análisis."
        else:
            # Calcular medidas de tendencia central utilizando la función en utils.py
            df, reporte_data, resultado = descriptiva_utils.calcular_tendencia_central(df, columnas_a_manipular)
            if resultado['Errores']:
                error_message = ', '.join(resultado['Errores'])
            else:
                # Guardar los resultados en la sesión para su posterior uso
                request.session['reporte_data'] = json.dumps(reporte_data)

                # Redirigir a la página del informe si no hay errores
                return redirect('report:reporte_tendencia_central', file_name=file_name)

    # Pasar las columnas al contexto del template incluso si hay un mensaje de error
    return render(request, 'data_analysis/tendencia_central.html', {
        'file_name': file_name,
        'columnas': columnas,
        'columnas_numericas': columnas_numericas,
        'dataframe': df_html,
        'error_message': error_message,
        "columna_seleccionada": columnas_a_manipular
    })

def variabilidad(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')

    # Obtener nombres de columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])

    # Variables para el mensaje de error
    error_message = None

    # Obtener las primeras 10 filas para las columnas seleccionadas
    columnas_a_manipular = request.session.get('columnas_a_manipular', [])
    primeras_filas = df[columnas_a_manipular].head(10).to_dict(orient='list') if columnas_a_manipular else {}

    # Manejo de POST
    if request.method == 'POST':
        columnas_a_manipular = request.POST.getlist('columnas_a_manipular')

        if not columnas_a_manipular:
            error_message = "No se especificaron columnas para el análisis."
        else:
            # Calcular medidas de variabilidad utilizando la función en utils.py
            df, reporte_data, resultado = descriptiva_utils.calcular_variabilidad(df, columnas_a_manipular)
            if resultado['Errores']:
                error_message = ', '.join(resultado['Errores'])
            else:
                # Guardar los resultados en la sesión para su posterior uso
                request.session['reporte_data'] = json.dumps(reporte_data)

                # Redirigir a la página del informe si no hay errores
                return redirect('report:reporte_variabilidad', file_name=file_name)

    #Pasar las columnas al contexto del template incluso si hay un mensaje de error
    return render(request, 'data_analysis/variabilidad.html', {
        'file_name': file_name,
        'columnas': columnas,
        'columnas_numericas': columnas_numericas,
        'dataframe': df_html,
        'error_message': error_message,
        "columna_seleccionada": columnas_a_manipular
    })


def regresion_lineal(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])

    #Manejo de POST
    if request.method == 'POST':
        variable_dependiente = request.POST.get('variable_dependiente')
        variables_independientes = request.POST.getlist('variables_independientes')
        
        # Verificar si se han seleccionado variables
        if not variable_dependiente or not variables_independientes:
            error_message = "Debe seleccionar al menos una variable dependiente y una variable independiente."
            return render(request, 'data_analysis/regresion_lineal.html', {
                'file_name': file_name,
                'columnas': columnas,
                'dataframe': df_html,
                'error_message': error_message
            })

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = inferencial_utils.regresion_linear(df, variable_dependiente, variables_independientes)
        
        if(resultado['Errores']):
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)

        request.session['reporte_data'] = json.dumps(reporte_data)

        guardar_csv(df_procesado, new_file_path)
        return redirect('report:reporte_regresion_linear', file_name=new_file_name)
    
    return render(request, 'data_analysis/regresion_lineal.html', {
        'file_name': file_name,
        'columnas': columnas,
        'dataframe': df_html
    })

def regresion_logistica(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])
    
    columnas_binarias = []
    columnas_numericas = df.select_dtypes(include=['float64', 'int', 'bool']).columns.tolist()

    for columna in df.columns:
        # Obtener valores únicos de la columna excluyendo NaN
        valores_unicos = pd.unique(df[columna].dropna())

        # Convertir valores únicos a strings para verificar si son numéricos
        valores_unicos_str = valores_unicos.astype(str)
        
        # Verificar si la columna tiene exactamente dos valores únicos y si son numéricos
        if len(valores_unicos) == 2 and all(valor.isnumeric() or valor.replace('.','',1).isdigit() for valor in valores_unicos_str):
            # convertir a float y verificar si esos valores son 0 y 1
            valores_unicos_float = valores_unicos.astype(float)
            if set(valores_unicos_float) == {0.0, 1.0}:
                columnas_binarias.append(columna)

    # Manejo de POST
    if request.method == 'POST':
        variable_dependiente = request.POST.get('variable_dependiente')
        variables_independientes = request.POST.getlist('variables_independientes')
        # Obtener el umbral del formulario, con un valor por defecto de 0.5 si no se proporciona
        umbral = float(request.POST.get('umbral', 0.5))
        
        # Verificar si se han seleccionado variables
        if not variable_dependiente or not variables_independientes:
            error_message = "Debe seleccionar al menos una variable dependiente y una variable independiente."
            return render(request, 'data_analysis/regresion_logistica.html', {
                'file_name': file_name,
                'columnas_numericas': columnas_numericas,
                'columnas_binarias': columnas_binarias,
                'dataframe': df_html,
                'error_message': error_message
            })

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = inferencial_utils.regresion_logistica(df, variable_dependiente, variables_independientes, umbral)
        
        if(resultado['Errores']):
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)

        request.session['reporte_data'] = json.dumps(reporte_data)

        guardar_csv(df_procesado, new_file_path)
        return redirect('report:reporte_regresion_logistica', file_name=new_file_name)
    
    return render(request, 'data_analysis/regresion_logistica.html', {
        'file_name': file_name,
        'columnas_numericas': columnas_numericas,
        'columnas_binarias': columnas_binarias,
        'dataframe': df.head(20).to_html(classes='table table', index=True)
    })

def correlacion_pearson(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])  # incluir solo columnas numéricas
    
    # Manejo de POST
    if request.method == 'POST':
        variables = request.POST.getlist('variables')  # Obtenemos las variables seleccionadas por el usuario
        
        # Verificar si no se han seleccionado suficientes variables
        if len(variables) < 2:
            error_message = "Debe seleccionar al menos dos variables para el análisis."
            return render(request, 'data_analysis/correlacion_pearson.html', {
                'file_name': file_name,
                'dataframe': df_html,
                'columnas': columnas,
                'error_message': error_message
            })

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = correlacion_utils.correlacion_pearson(df, variables)
        
        if resultado['Errores']:
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)
        
        request.session['reporte_data'] = json.dumps(reporte_data)
        
        guardar_csv(df_procesado, new_file_path)
        return redirect('report:reporte_correlacion_pearson', file_name=new_file_name)
    
    return render(request, 'data_analysis/correlacion_pearson.html', {
        'file_name': file_name,
        'columnas': columnas,
        'dataframe': df_html
    })

def correlacion_spearman(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])

    # Manejo de POST
    if request.method == 'POST':
        variables = request.POST.getlist('variables')  # Obtenemos las variables seleccionadas por el usuario
        
        # Verificar si no se han seleccionado suficientes variables
        if len(variables) < 2:
            error_message = "Debe seleccionar al menos dos variables para el análisis."
            return render(request, 'data_analysis/correlacion_spearman.html', {
                'file_name': file_name,
                'dataframe': df_html,
                'columnas': columnas,
                'error_message': error_message
            })


        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = correlacion_utils.correlacion_spearman(df, variables)
        
        if resultado['Errores']:
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)
        
        request.session['reporte_data'] = json.dumps(reporte_data)  # Guardamos los datos del reporte en la sesión
        
        guardar_csv(df_procesado, new_file_path)  # Guardamos el DataFrame procesado
        return redirect('report:reporte_correlacion_spearman', file_name=new_file_name)  # Redirigimos al reporte de correlación de Spearman

    return render(request, 'data_analysis/correlacion_spearman.html', {
        'file_name': file_name,
        'columnas': columnas,
        'dataframe': df_html
    })

def kmeans_clustering(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])  # Mostrar solo columnas numéricas

    # Manejo de POST
    if request.method == 'POST':
        caracteristicas = request.POST.getlist('caracteristicas')
        n_clusters = request.POST.get('n_clusters', 3)

        # Verificar si se han seleccionado variables
        if not caracteristicas or len(caracteristicas) < 2:
            error_message = "Debe seleccionar al menos dos caracteristicas."
            return render(request, 'data_analysis/kmeans_clustering.html', {
                'file_name': file_name,
                'columnas': columnas,
                'dataframe': df_html,
                'error_message': error_message
            })

        # Convertir el parámetro n_clusters recibido como string a int
        n_clusters = int(n_clusters) if n_clusters and n_clusters.isdigit() else 3

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = machine_learning_utils.kmeans_clustering(
            df, caracteristicas, n_clusters
        )

        if resultado['Errores']:
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)

        request.session['reporte_data'] = json.dumps(reporte_data)  # Guardamos los datos del reporte en la sesión
        
        guardar_csv(df_procesado, new_file_path)  # Guardamos el DataFrame procesado
        return redirect('report:reporte_kmeans_clustering', file_name=new_file_name)  # Redirigimos al reporte de K-means clustering

    return render(request, 'data_analysis/kmeans_clustering.html', {
        'file_name': file_name,
        'columnas': columnas,
        'dataframe': df_html
    })


def arbol_decision_clasificacion(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number', 'category_number'])  # Mostrar solo columnas numéricas
    
    # Manejo de POST
    if request.method == 'POST':
        variable_objetivo = request.POST.get('variable_objetivo')
        caracteristicas = request.POST.getlist('caracteristicas')
        max_depth = request.POST.get('max_depth', None)
        min_samples_leaf = request.POST.get('min_samples_leaf', 1)

        # Verificar si se han seleccionado variables
        if not variable_objetivo or not caracteristicas or len(caracteristicas) < 1:
            error_message = "Debe seleccionar una variable objetivo (y) y al menos una característica (x)."
            return render(request, 'data_analysis/arbol_decision_clasificacion.html', {
                'file_name': file_name,
                'columnas_numericas': columnas['number'],
                'columnas_categoricas': columnas['category_number'],
                'dataframe': df_html,
                'error_message': error_message
            })

        # Convertir los parámetros numéricos recibidos como string a int
        max_depth = int(max_depth) if max_depth and max_depth.isdigit() else None
        min_samples_leaf = int(min_samples_leaf) if min_samples_leaf and min_samples_leaf.isdigit() else 1

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = machine_learning_utils.arbol_decision_clasificacion(
            df, caracteristicas, variable_objetivo, max_depth, min_samples_leaf
        )
        
        if resultado['Errores']:
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)
        
        request.session['reporte_data'] = json.dumps(reporte_data)  # Guardamos los datos del reporte en la sesión
        
        guardar_csv(df_procesado, new_file_path)  # Guardamos el DataFrame procesado
        return redirect('report:reporte_arbol_decision_clasificacion', file_name=new_file_name)  # Redirigimos al reporte del árbol de decisión

    return render(request, 'data_analysis/arbol_decision_clasificacion.html', {
        'file_name': file_name,
        'columnas_numericas': columnas['number'],
        'columnas_categoricas': columnas['category_number'],
        'dataframe': df_html
    })





def intervalo(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas_numericas = preparar_datos_para_get(df, include_dtypes=['number'])
    
    # Manejo de POST
    if request.method == 'POST':
        columnas_a_analizar = request.POST.getlist('columnas_a_manipular')
        
        # Verificar si se han seleccionado columnas para el análisis
        if not columnas_a_analizar:
            mensaje_error = "No se especificaron columnas para el análisis."
            return render(request, 'data_analysis/intervalo.html', {
                'file_name': file_name,
                'columnas_numericas': columnas_numericas,
                'dataframe': df_html,
                'error_message': mensaje_error
            })

        # Calcular intervalo de confianza para cada columna seleccionada
        df, reporte_data, resultado = inferencial_utils.calcular_intervalo_confianza(df, columnas_a_analizar)
        
        if resultado['Errores']:
            return render(request, 'data_analysis/intervalo.html', {
                'file_name': file_name,
                'columnas_numericas': columnas_numericas,
                'dataframe': df_html,
                'error_message': ' '.join(resultado['Errores'])
            })

        request.session['reporte_data'] = json.dumps(reporte_data, default=str)  # Guardar reporte_data en la sesión

        # Guardar el DataFrame modificado (si es necesario)
        guardar_csv(df, file_path)

        # Redirigir a la página de resultados
        return redirect('report:reporte_intervalo_confianza', file_name=file_name)
    
    return render(request, 'data_analysis/intervalo.html', {
        'file_name': file_name,
        'columnas_numericas': columnas_numericas,
        'dataframe': df_html
    })



def prueba_hipotesis(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')
    
    # Preparación para el método GET
    df_html, columnas_numericas = preparar_datos_para_get(df, include_dtypes=['number'])  # Asegúrate de definir esta función

    # Manejo de POST
    if request.method == 'POST':
        columnas_a_analizar = request.POST.getlist('columnas_a_manipular')
        mu = float(request.POST.get('mu'))
        tipo_prueba = request.POST.get('tipo_prueba')
        nivel_significancia = float(request.POST.get('nivel_significancia'))

        # Verificar si se han seleccionado columnas para el análisis
        if not columnas_a_analizar:
            mensaje_error = "No se especificaron columnas para el análisis."
            return render(request, 'data_analysis/prueba_hipotesis.html', {
                'file_name': file_name,
                'columnas_numericas': columnas_numericas,
                'dataframe': df_html,
                'error_message': mensaje_error
            })

        # Ajustar la función de análisis para manejar mu, tipo de prueba y nivel de significancia
        df, reporte_data, resultado = inferencial_utils.prueba_t(df, columnas_a_analizar, mu, tipo_prueba, nivel_significancia)
        
        if resultado['Errores']:
            return render(request, 'data_analysis/prueba_hipotesis.html', {
                'file_name': file_name,
                'columnas_numericas': columnas_numericas,
                'dataframe': df_html,
                'error_message': ' '.join(resultado['Errores'])
            })

        request.session['reporte_data'] = json.dumps(reporte_data, default=str)  # Guardar reporte_data en la sesión

        # Guardar el DataFrame modificado (si es necesario)
        guardar_csv(df, file_path)

        # Redirigir a la página de resultados
        return redirect('report:reporte_prueba_hipotesis', file_name=file_name)
    
    return render(request, 'data_analysis/prueba_hipotesis.html', {
        'file_name': file_name,
        'columnas_numericas': columnas_numericas,
        'dataframe': df_html
    })


def perceptron(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')

    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number', 'int'])  # Mostrar solo columnas numéricas
    

    # Manejo de POST
    if request.method == 'POST':
        variable_objetivo = request.POST.get('variable_objetivo')
        caracteristicas = request.POST.getlist('caracteristicas')
        hidden_layer_sizes_str = request.POST.get('hidden_layer_sizes', '(100)')
        activation = request.POST.get('activation', 'relu')
        solver = request.POST.get('solver', 'adam')
        alpha = float(request.POST.get('alpha', 0.0001))

        # Verificar si se han seleccionado variables y que hidden_layer_sizes tiene un formato válido
        if not variable_objetivo or not caracteristicas or len(caracteristicas) < 1 or not re.match(r'^\d+(,\d+)*$', hidden_layer_sizes_str):
            error_message = "Debe seleccionar una variable objetivo (y) y al menos una característica (x). Verifique también el formato de 'Tamaños de las capas ocultas' (sólo números, separados por comas)."
            return render(request, 'data_analysis/perceptron.html', {
                'file_name': file_name,
                'columnas_x': columnas['number'],
                'columnas_y': columnas['int'],
                'dataframe': df_html,
                'error_message': error_message
            })

        # Convertir hidden_layer_sizes a tupla de enteros
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes_str.strip('()').split(',')))

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        df_procesado, reporte_data, resultado = machine_learning_utils.perceptron_multicapa_clasificacion(
            df, caracteristicas, variable_objetivo, hidden_layer_sizes, activation, solver, alpha
        )
        
        if resultado['Errores']:
            guardar_resultado_en_sesion(request, resultado)
            return redirect('file_handler:revisar_csv', file_name=new_file_name)
        
        request.session['reporte_data'] = json.dumps(reporte_data)  # Guardamos los datos del reporte en la sesión
        
        guardar_csv(df_procesado, new_file_path)  # Guardamos el DataFrame procesado
        return redirect('report:reporte_perceptron', file_name=new_file_name)  # Redirigimos al reporte de perceptrón multicapa

    return render(request, 'data_analysis/perceptron.html', {
        'file_name': file_name,
        'columnas_x': columnas['number'],
        'columnas_y': columnas['int'],
        'dataframe': df_html
    })