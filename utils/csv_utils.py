import datetime
import json
import os
import shutil
import uuid
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect
import pandas as pd

# Este script está dedicado a la manipulación de archivos CSV, se encarga de cargar, revisar y descargar archivos CSV
# Es crucial para el funcionamiento de la aplicación.
# Se deberían revisar las funciones de lectura y paso de csvs entre vistas, ya que un csv puede quedar vacío y esto puede causar errores en la aplicación, 
# pues asume que siempre habrá un csv cargado.

# Se incluyen las funciones: handle_uploaded_file, leer_csv_o_error, leer_y_verificar_csv, manejar_error_csv, guardar_csv, 
# gestionar_version_archivo, crear_copia_archivo y registrar_version_json_multiple

def handle_uploaded_file(f, session_key):
    session_dir = os.path.join(settings.TEMP_FILES_DIR, f"session_{session_key}")
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    unique_id = str(uuid.uuid4())
    unique_filename = f"{unique_id}.csv"
    file_path = os.path.join(session_dir, unique_filename)
    
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    
    registrar_version_json_multiple(session_dir, unique_id, unique_filename)
    
    return unique_filename

def leer_csv_o_error(request, file_name):
    session_key = request.session.session_key
    session_dir = os.path.join(settings.TEMP_FILES_DIR, f"session_{session_key}")
    file_path = os.path.join(session_dir, file_name)

    df, error = leer_y_verificar_csv(file_path)
    if error:
        manejar_error_csv(request, error, file_name)
        return None, error, None
    return df, None, file_path

def leer_y_verificar_csv(file_path):
    if not os.path.exists(file_path):
        return None, "Archivo no encontrado."
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return None, "CSV vacío"
    except Exception as e:
        return None, str(e)

    return df, None

def manejar_error_csv(request, error, file_name):
    if error == "CSV vacío":
        request.session['csv_vacio_mensaje'] = 'El archivo CSV está vacío. Por favor, carga un nuevo archivo.'
    else:
        request.session['csv_vacio_mensaje'] = 'El archivo CSV está vacío u ocurrió otro error inesperado. Por favor, carga un nuevo archivo.'

def guardar_csv(df, file_path):
    df.to_csv(file_path, index=False)




def gestionar_version_archivo(request, file_name):
    session_key = request.session.session_key
    session_dir = os.path.join(settings.TEMP_FILES_DIR, f"session_{session_key}")

    original_id = file_name.split('.')[0]

    new_file_name, new_file_path = crear_copia_archivo(original_id, file_name, session_dir)

    return new_file_name, new_file_path

def crear_copia_archivo(original_id, file_name, session_dir):
    original_file_path = os.path.join(session_dir, file_name)
    if not os.path.exists(original_file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {original_file_path}")

    # Extrae identificador base sin los sufijos de versión
    base_id = original_id.split('_')[0]  # Se asume que original_id puede tener sufijos

    # +1 el número de versión basándonos en los archivos existentes
    existing_files = [f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f)) and f.startswith(base_id)]
    version_number = len(existing_files)  # asigna un nuevo número de versión (++)

    new_file_name = f"{base_id}_{version_number}.csv"
    new_file_path = os.path.join(session_dir, new_file_name)
    shutil.copy2(original_file_path, new_file_path)
    
    # Registra la nueva versión usando siempre el base_id como clave
    registrar_version_json_multiple(session_dir, base_id, new_file_name)
    
    return new_file_name, new_file_path

def registrar_version_json_multiple(session_dir, base_id, new_file_name):
    metadata_path = os.path.join(session_dir, 'metadata.json')
    new_entry = {
        'nombre_archivo': new_file_name,
        'ruta': os.path.join(session_dir, new_file_name),
        'timestamp': datetime.datetime.now().isoformat()
    }

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as file:
            data = json.load(file)
        if base_id in data:
            data[base_id].append(new_entry)
        else:
            data[base_id] = [new_entry]
    else:
        data = {base_id: [new_entry]}

    with open(metadata_path, 'w') as file:
        json.dump(data, file, indent=4)