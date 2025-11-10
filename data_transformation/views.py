from django.shortcuts import render
from django.shortcuts import redirect, render
from .utils import standardization_utils, transformation_utils
from .utils.views_utils import guardar_resultado_en_sesion, preparar_datos_para_get
from utils.csv_utils import gestionar_version_archivo, leer_csv_o_error, guardar_csv

# Vistas de transformación de datos y estandarización.
# Se encargan de recibir los datos del usuario, procesarlos y enviarlos a la vista de revisión de datos.
# Se incluyen las vistas: transformation y standardization

def transformation(request, file_name):
    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')


    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['category'])

    unique_values = {col: df[col].dropna().unique().tolist() for col in columnas}
    # Se cre un nuevo diccionario para el contexto que contenga tanto las columnas como sus valores únicos
    column_info = {col: {'values': unique_values[col]} for col in columnas if col in unique_values}
    orders = {}


    if request.method == 'POST':
        tipo = request.POST.get('tipo', '')
        columnas_a_manipular = request.POST.getlist('columnas_a_manipular')

        if not columnas_a_manipular:
            error_message = "Debe seleccionar al menos una columna."
            return render(request, 'data_transformation/transformation.html', {
                'file_name': file_name,
                'dataframe': df_html,
                'columnas': columnas,
                'column_info': column_info,
                'orders': orders,
                'error_message': error_message
            })

        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)
        
        # Recolección de los órdenes para el codificador ordinal
        orders = {}
        for columna in columnas_a_manipular:
            if tipo == 'ordinalencoder':
                order_key = f'ordinal_order_{columna}'
                orders[columna] = request.POST.get(order_key, '').split(',')

        df_procesado, resultado = transformation_utils.transformationStand(df, tipo, columnas_a_manipular, orders=orders)

        guardar_resultado_en_sesion(request, resultado)
        guardar_csv(df_procesado, new_file_path)
        return redirect('file_handler:revisar_csv', file_name=new_file_name)

    return render(request, 'data_transformation/transformation.html', {
        'file_name': file_name,
        'columnas': columnas,
        'column_info': column_info,
        'orders': orders,
        'dataframe': df_html
    })

def standardization(request, file_name):

    # Manejo de error y carga de DataFrame
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        return redirect('file_handler:cargar_archivo')

    # Preparación para el método GET
    df_html, columnas = preparar_datos_para_get(df, include_dtypes=['number'])


    # Manejo de POST
    if request.method == 'POST':
        tipo = request.POST.get('tipo','')
        columnas_a_manipular = request.POST.getlist('columnas_a_manipular')
        
        if not columnas_a_manipular:
            error_message = "Debe seleccionar al menos una columna."
            return render(request, 'data_transformation/standardization.html', {
                'file_name': file_name,
                'dataframe': df_html,
                'columnas': columnas,
                'error_message': error_message
            })
        
        new_file_name, new_file_path = gestionar_version_archivo(request, file_name)

        df_procesado, resultado = standardization_utils.standardization(df, tipo, columnas_a_manipular)

        guardar_resultado_en_sesion(request, resultado)
        guardar_csv(df_procesado, new_file_path)
        return redirect('file_handler:revisar_csv', file_name=new_file_name)


    return render(request, 'data_transformation/standardization.html', {
        'file_name': file_name,
        'columnas': columnas, 
        'dataframe': df_html
    })