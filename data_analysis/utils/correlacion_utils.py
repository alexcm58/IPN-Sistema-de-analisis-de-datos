import uuid
import pandas as pd
import numpy as np
from django.conf import settings
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# Este script está dedicado a la creación de funciones que permiten realizar análisis de correlación de Pearson y Spearman, 
# crear visualizaciones y envair los resultados obtenidos por medio de un diccionario.
# Incluye las funciones correlacion_pearson y correlacion_spearman.

def correlacion_pearson(df, variables):
    resultado = {
        'Errores': []
    }

    if len(variables) < 2:
        resultado['Errores'].append("Se requieren al menos dos variables para analizar la correlación.")
        return df, None, resultado
    
    try:
        primeras_filas = df[variables].head(10)
        matriz_correlacion = df[variables].corr(method='pearson')

        sns.set(style="whitegrid")  
        sns.set_context("talk", font_scale=0.6) 

        # Generar el heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap='RdBu_r',
                    xticklabels=variables, yticklabels=variables, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.tight_layout()

        unique_filename = str(uuid.uuid4()) + '.png'
        heatmap_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename)
        plt.savefig(heatmap_path, dpi=400)
        plt.close()

        # Datos del reporte
        reporte_data = {
            "titulo": "Reporte - Análisis de Correlación de Pearson",
            "descripcion": f"Análisis de correlación de Pearson para evaluar la relación lineal entre las variables:  [{', '.join(variables)}].",
            "tipo_analisis": "correlacion_pearson",
            "resultados": {
                "nombres_columnas": variables,
                "primeras_filas": primeras_filas.to_dict('records'),
                "matriz_correlacion": matriz_correlacion.to_dict(),
                "heatmap_path": unique_filename 
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None
    
    return df, reporte_data, resultado



def correlacion_spearman(df, variables):
    resultado = {
        'Errores': []
    }

    if len(variables) < 2:
        resultado['Errores'].append("Se requieren al menos dos variables para analizar la correlación.")
        return df, None, resultado
    
    try:
        primeras_filas = df[variables].head(10)
        matriz_correlacion = df[variables].corr(method='spearman')

        # Generar el heatmap
        sns.set(style="whitegrid") 
        sns.set_context("talk", font_scale=0.6)

        matriz_correlacion = df[variables].corr(method='spearman')

        # Plot el heatmap
        plt.figure(figsize=(10, 8))

        # paleta de colores
        sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap='vlag',
                    xticklabels=variables, yticklabels=variables, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})

        plt.tight_layout()

        # creacion de ruta del archivo
        unique_filename = str(uuid.uuid4()) + '.png' 
        heatmap_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename)
        plt.savefig(heatmap_path, dpi=400)
        plt.close()

        # Añadir la ruta del archivo al reporte_data
        reporte_data = {
            "titulo": "Reporte - Análisis de Correlación de Spearman",
            "descripcion": f"Análisis de correlación de Spearman para evaluar la relación entre las variables: [{', '.join(variables)}].",
            "tipo_analisis": "correlacion_spearman",
            "resultados": {
                "nombres_columnas": variables,
                "primeras_filas": primeras_filas.to_dict('records'),
                "matriz_correlacion": matriz_correlacion.to_dict(),
                "heatmap_path": unique_filename  # Guardar solo el nombre del archivo para mayor flexibilidad
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None
    
    return df, reporte_data, resultado

