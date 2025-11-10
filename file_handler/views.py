
import csv
import json
import os
from urllib.parse import urlencode
from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseRedirect
from http.client import HTTPResponse
from django.http import JsonResponse
from django.shortcuts import render, redirect
import pandas as pd
from .forms import CargaCSVForm
from utils.tablas_utils import crear_inicio_tabla
from utils.csv_utils import leer_csv_o_error, handle_uploaded_file

# Vistas dedicadas a la manipulación de archivos CSV, se encargan de cargar, revisar y descargar archivos CSV
# Se incluyen las vistas: cargar_archivo, revisar_csv, descargar_archivo_csv, cargar_mas_filas y cambiar_version

def cargar_archivo(request):
    mensaje_error = request.session.get('csv_vacio_mensaje', None)
    request.session['csv_vacio_mensaje'] = None

    if request.method == 'POST':
        form = CargaCSVForm(request.POST, request.FILES)
        if form.is_valid():
            archivo_csv = request.FILES['archivo_csv']
            unique_file_name = handle_uploaded_file(archivo_csv, request.session.session_key)
            return redirect('file_handler:revisar_csv', file_name=unique_file_name)
    else:
        form = CargaCSVForm()
    return render(request, 'file_handler/cargar_archivo.html', {
        'form': form,
        'mensaje_error': mensaje_error 
    })

def revisar_csv(request, file_name):
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    if error_response:
        # Devuelve una respuesta HTTP adecuada para errores
        return error_response
    
    info = request.session.get('info', None) # Obtiene el mensaje de éxito o error
    request.session['info'] = None  # Borra el mensaje de la sesión

    return render(request, 'file_handler/revisar_csv.html', {
        'dataframe': crear_inicio_tabla(df),
        'file_name': file_name,
        'info': info
    })


def descargar_archivo_csv(request, file_name):
    # Leer el archivo CSV subido por el usuario
    df, error_response, file_path = leer_csv_o_error(request, file_name)
    
    # Verificar si hubo un error al leer el archivo CSV
    if error_response:
        return error_response
    
    # Construir la respuesta HTTP para la descarga del archivo CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'  # Nombre del archivo
    
    # Escribir el contenido del archivo CSV en la respuesta HTTP
    writer = csv.writer(response)
    
    # Escribir el encabezado (nombres de las columnas)
    writer.writerow(df.columns)
    
    # Escribir los datos (filas)
    for index, row in df.iterrows():
        writer.writerow(row)
    
    return response




########################################################################################################

# Cargar más filas
def cargar_mas_filas(request, file_name):
    df, error_response, _ = leer_csv_o_error(request, file_name)
    if error_response:
        return error_response

    start_row = int(request.GET.get('start', 0))
    df_partial = df.iloc[start_row:start_row + 20]
    df_html = df_partial.to_html(classes='table table', index=True, header=False)
    return JsonResponse({'data': df_html})


########################################################################################################

# cambiar versiones
def cambiar_version(request, file_name, direction):
    session_key = request.session.session_key
    try:
        new_file_name, _ = get_new_version(session_key, file_name, direction)
        if new_file_name:
            # Obtén la URL referente y reemplaza el nombre antiguo del archivo con el nuevo
            referer_url = request.META.get('HTTP_REFERER', '/')
            # Suponiendo que el nombre del archivo está al final de la URL:
            base_url = referer_url.rsplit('/', 2)[0]
            new_referer_url = f"{base_url}/{new_file_name}/"
            return HttpResponseRedirect(new_referer_url)
        else:
            # Si no hay más versiones, redirige a la URL referente sin cambios
            return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
    except Exception as e:
        # Si hay un error, loguea y redirige a la URL referente o a la raíz
        print(f"Error cambiando la versión: {str(e)}")
        return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
    
# def cambiar_version(request, file_name, direction):
#     session_key = request.session.session_key
#     try:
#         new_file_name, _ = get_new_version(session_key, file_name, direction)
#         if new_file_name:
#             return redirect('file_handler:revisar_csv', file_name=new_file_name)
#         else:
#             return redirect('file_handler:revisar_csv', file_name=file_name)
#     except Exception as e:
#         print(f"Error cambiando la versión: {str(e)}")
#         return redirect('file_handler:revisar_csv', file_name=file_name)

def get_new_version(session_key, current_file_name, direction):
    session_dir = os.path.join(settings.TEMP_FILES_DIR, f"session_{session_key}")
    metadata_path = os.path.join(session_dir, 'metadata.json')

    if not os.path.exists(metadata_path):
        return None, None  # Si no existe metadata, no se hace nada

    with open(metadata_path, 'r') as file:
        data = json.load(file)

    # Encontrar la lista de versiones para el archivo original
    for original_file_name, versions in data.items():
        for index, entry in enumerate(versions):
            if entry['nombre_archivo'] == current_file_name:
                # Determinar la nueva versión basada en la dirección
                if direction == 'next' and index < len(versions) - 1:
                    new_version = versions[index + 1]
                elif direction == 'prev' and index > 0:
                    new_version = versions[index - 1]
                else:
                    return None, None  # No hay versión anterior/siguiente

                # Cargar el DataFrame de la nueva versión
                new_file_path = new_version['ruta']
                try:
                    df = pd.read_csv(new_file_path)
                    dataframe_html = df.head(20).to_html(classes='table table', index=True)
                    print("Nueva versión encontrada: ", new_version['nombre_archivo'])  # Log para depuración
                    return new_version['nombre_archivo'], dataframe_html
                except Exception as e:
                    print(f"Error loading CSV: {e}")
                    return None, None

    return None, None  # Si no se encuentra el archivo en la lista