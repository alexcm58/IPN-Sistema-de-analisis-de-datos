import numpy as np
import pandas as pd

# Las funciones de este script se utilizan para realizar análisis descriptivo de datos y se usan en las views de análisis de datos.
# Obtiene información sobre la distribución de frecuencias, tendencia central y variabilidad de los datos y crea un reporte con los resultados.
# Incluye las funciones calcular_distribucion_frecuencias, calcular_tendencia_central y calcular_variabilidad.


def calcular_distribucion_frecuencias(df, columnas):
    resultado = {
        'Errores': []
    }
    reporte_data = None

    try:
        # Calcular distribución de frecuencias para cada columna
        resultados_distribucion_frecuencias = {}
        for columna in columnas:
            frecuencia = df[columna].value_counts().to_dict()
            resultados_distribucion_frecuencias[columna] = frecuencia

        # Crear datos del reporte
        reporte_data = {
            "titulo": "Reporte - Análisis de Distribución de Frecuencias",
            "descripcion": f"Análisis de la distribución de frecuencias realizado sobre las columnas: [{', '.join(columnas)}].",
            "tipo_analisis": "distribucion_frecuencias",
            "resultados": {
                "nombres_columnas": columnas,
                "primeras_filas": df[columnas].head(10).to_dict('records'),
                "frecuencias": resultados_distribucion_frecuencias
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
    
    return df, reporte_data, resultado


def calcular_tendencia_central(df, columnas):
    resultado = {
        'Errores': []
    }
    reporte_data = None

    try:
        # Calcular medidas de tendencia central para cada columna
        resultados_tendencia_central = {}
        for columna in columnas:
            media = df[columna].mean()
            moda = df[columna].mode()[0]
            mediana = df[columna].median()

            resultados_tendencia_central[columna] = {
                'Media': media.item() if isinstance(media, (np.integer, np.floating)) else media,
                'Moda': moda.item() if isinstance(moda, (np.integer, np.floating)) else moda,
                'Mediana': mediana.item() if isinstance(mediana, (np.integer, np.floating)) else mediana
            }

        # Convertir primeras filas a tipos nativos de Python
        primeras_filas = df[columnas].head(10).to_dict('records')
        primeras_filas = [{k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in fila.items()} for fila in primeras_filas]

        # Crear datos del reporte
        reporte_data = {
            "titulo": "Reporte - Análisis de Tendencia Central",
            "descripcion": f"Análisis de la tendencia central realizado sobre las columnas: [{', '.join(columnas)}].",
            "tipo_analisis": "tendencia_central",
            "resultados": {
                "nombres_columnas": columnas,
                "primeras_filas": primeras_filas,
                "tendencia_central": resultados_tendencia_central
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
    
    return df, reporte_data, resultado



def calcular_variabilidad(df, columnas):
    resultado = {
        'Errores': []
    }
    reporte_data = None

    try:
        # Calcular medidas de variabilidad para cada columna
        resultados_variabilidad = {}
        for columna in columnas:
            varianza = df[columna].var()
            desviacion_std = df[columna].std()
            rango = df[columna].max() - df[columna].min()
            coef_variacion = desviacion_std / df[columna].mean() if df[columna].mean() != 0 else np.nan

            resultados_variabilidad[columna] = {
                'Varianza': varianza.item() if isinstance(varianza, (np.integer, np.floating)) else varianza,
                'Desviacion_Estandar': desviacion_std.item() if isinstance(desviacion_std, (np.integer, np.floating)) else desviacion_std,
                'Rango': rango.item() if isinstance(rango, (np.integer, np.floating)) else rango,
                'Coeficiente_de_Variacion': coef_variacion.item() if isinstance(coef_variacion, (np.integer, np.floating)) else coef_variacion
            }

        # Convertir primeras filas a tipos nativos de Python
        primeras_filas = df[columnas].head(10).to_dict('records')
        primeras_filas = [{k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in fila.items()} for fila in primeras_filas]

        # Crear datos del reporte
        reporte_data = {
            "titulo": "Reporte - Análisis de Variabilidad",
            "descripcion": f"Análisis de la variabilidad realizado sobre las columnas: [{', '.join(columnas)}].",
            "tipo_analisis": "variabilidad",
            "resultados": {
                "nombres_columnas": columnas,
                "primeras_filas": primeras_filas,
                "variabilidad": resultados_variabilidad
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
    
    return df, reporte_data, resultado
