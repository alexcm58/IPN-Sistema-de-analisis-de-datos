import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import os
import uuid
from django.conf import settings
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
matplotlib.use('agg')

# Este script contiene funciones que permiten realizar análisis de Machine Learning, como clustering y clasificación, y envía los resultados obtenidos por medio de un diccionario.
# Incluye las funciones kmeans_clustering, arbol_decision_clasificacion y perceptron_multicapa_clasificacion.


def kmeans_clustering(df, columnas, n_clusters):
    resultado = {
        'Errores': []
    }
    
    # Comprobaciones iniciales
    if not set(columnas).issubset(df.columns):
        resultado['Errores'].append("Alguna de las columnas especificadas no existe en el DataFrame.")
        return df, None, resultado

    if df[columnas].isnull().any().any():
        resultado['Errores'].append("Hay valores nulos en las columnas seleccionadas.")
        return df, None, resultado

    try:
        # Datos para el modelo
        X = df[columnas]

        # Obtener las primeras 10 filas para las columnas especificadas
        primeras_filas = df[columnas].head(10)
        
        # Entrenamiento del modelo K-means
        modelo = KMeans(n_clusters=n_clusters, random_state=42)
        modelo.fit(X)

        # Asignación de etiquetas y centroides
        etiquetas = modelo.labels_
        centroides = modelo.cluster_centers_

        # Preparar etiquetas con indices
        # etiquetas_con_indices = list(enumerate(etiquetas.tolist()))

        # Añadiendo la columna de cluster para visualización
        df['Cluster'] = etiquetas

        # Calcula la distribución de las muestras por cada cluster utilizando el arreglo de etiquetas
        etiquetas_series = pd.Series(etiquetas)
        distribucion_clusters = etiquetas_series.value_counts().to_dict()

        # Cálculo de Silhouette Score para evaluar la calidad de los clusters
        silhouette = silhouette_score(X, etiquetas)
        silhouette_values = silhouette_samples(X, etiquetas)

        # Configurar el gráfico
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        y_lower = 10
        color_palette = plt.cm.get_cmap("flare", n_clusters)
        
        for i in range(n_clusters):
            ith_cluster_silhouette_values = silhouette_values[etiquetas == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = color_palette(i)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Etiqueta de índice del cluster
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            y_lower = y_upper + 10 

        ax1.set_title('Gráfico de Silueta para los diversos clusters')
        ax1.set_xlabel('Valores del coeficiente de silueta')
        ax1.set_ylabel('Etiqueta del cluster')

        # línea vertical para el promedio de todos los valores de silueta
        silhouette_avg = np.mean(silhouette_values)
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([]) 
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Guardar la imagen
        unique_filename_silhouette = str(uuid.uuid4()) + '_silhouette.png'
        silhouette_image_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename_silhouette)
        plt.savefig(silhouette_image_path, dpi=400)
        plt.close()


        # Configurar el gráfico pair plot
        plt.figure(figsize=(10, 6))

        # Crear el Pair Plot usando Seaborn
        sns.pairplot(df, hue='Cluster', vars=columnas, palette='flare')

        # Guardar la imagen del gráfico
        unique_filename = str(uuid.uuid4()) + '.png'
        image_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename)
        plt.savefig(image_path, dpi=400)
        plt.close()

        # Eliminar la columna 'Cluster' para mantener el DataFrame original sin cambios
        df.drop('Cluster', axis=1, inplace=True)


        # Creación del reporte
        reporte_data = {
            "titulo": "Reporte - K-Means Clustering",
            "descripcion": f"Resultado de K-means clustering realizado sobre las variables: [{', '.join(columnas)}].",
            "tipo_analisis": "kmeans",
            "resultados": {
                "nombres_columnas": columnas,
                "centroides": centroides.tolist(),
                "distribucion_muestras": distribucion_clusters,
                "path_imagen_clusters": unique_filename,
                "path_imagen_silhouette": unique_filename_silhouette,
                # "etiquetas_clusters": etiquetas_con_indices,
                "primeras_filas": primeras_filas.to_dict('records'),
                "estadisticas": {
                    "Silhouette Score": silhouette,
                    "Número de Clusters": n_clusters,
                    "Método de Inicialización": modelo.init,
                    "Número de Iteraciones": modelo.n_iter_
                }
            }
        }
    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None
    
    return df, reporte_data, resultado


def arbol_decision_clasificacion(df, columnas_x, columna_y, max_depth=None, min_samples_leaf=1):
    resultado = {
        'Errores': []
    }
    
    # Comprobaciones iniciales
    if not set(columnas_x).issubset(df.columns) or columna_y not in df.columns:
        resultado['Errores'].append("Alguna de las columnas especificadas no existe en el DataFrame.")
        return df, None, resultado
    
    if df[columnas_x].isnull().any().any() or df[columna_y].isnull().any():
        resultado['Errores'].append("Hay valores nulos en las columnas seleccionadas.")
        return df, None, resultado
    
    # División de los datos en conjunto de entrenamiento y prueba
    X = df[columnas_x]
    y = df[columna_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    try:
        # Obtener las primeras 10 filas para las columnas especificadas
        columnas_unicas = list(set(columnas_x + [columna_y]))
        primeras_filas = df[columnas_unicas].head(10)
    
        # Entrenamiento del modelo
        modelo = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        modelo.fit(X_train, y_train)
        
        # Predicción y cálculo del accuracy score
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        # Transformación del reporte de clasificación para la tabla
        reporte_clasificacion = []
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                reporte_clasificacion.append({
                    'clase': label,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1-score'],
                    'soporte': metrics['support']
                })

        feature_importances = modelo.feature_importances_
        # Mapeo de las importancias a las columnas correspondientes
        importancias_caracteristicas = dict(zip(columnas_x, feature_importances))
        
        # Visualización del árbol de decisión
        plt.figure(figsize=(20,10))
        class_names = [str(x) for x in modelo.classes_]
        plot_tree(modelo, filled=True, feature_names=columnas_x, class_names=class_names)
        plt.title('Árbol de Decisión de Clasificación')

        # Guardar la imagen del árbol
        unique_filename = str(uuid.uuid4()) + '.png'
        tree_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename)
        plt.savefig(tree_path, dpi=400)
        plt.close()

        class_names = [str(x) for x in modelo.classes_]

        # Creación del reporte
        reporte_data = {
            "titulo": "Reporte - Árbol de Decisión de Clasificación",
            "descripcion": f"Resultado de Árbol de decisión utilizado para clasificar con variable objetivo (Y):  [{columna_y}] y las características: (X): [{', '.join(columnas_x)}].",
            "tipo_analisis": "arbol_decision",
            "resultados": {
                "nombres_columnas": columnas_unicas,
                "classes": class_names,
                "matriz_confusion": conf_matrix.tolist(),
                "reporte_clasificacion": reporte_clasificacion,
                "importancia_caracteristicas": importancias_caracteristicas,
                "primeras_filas": primeras_filas.to_dict('records'),
                "path_imagen_arbol": unique_filename,
                "estadisticas": {
                    "Accuracy": accuracy,
                    "Max Depth": max_depth,
                    "Min Samples Leaf": min_samples_leaf,
                    "Variable Objetivo": columna_y,
                    "Variables Predictoras": columnas_x
                }
            }
        }
        
        
        reporte_data['resultados']['matriz_confusion_indexada'] = list(enumerate(reporte_data['resultados']['matriz_confusion']))
        clases_y_filas = zip(reporte_data['resultados']['classes'], reporte_data['resultados']['matriz_confusion'])
        reporte_data['resultados']['clases_y_filas'] = list(clases_y_filas)

    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None
    
    return df, reporte_data, resultado



def perceptron_multicapa_clasificacion(df, columnas_x, columna_y, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001):
    resultado = {
        'Errores': []
    }
    
    # Comprobaciones iniciales
    if not set(columnas_x).issubset(df.columns) or columna_y not in df.columns:
        resultado['Errores'].append("Alguna de las columnas especificadas no existe en el DataFrame.")
        return df, None, resultado
    
    if df[columnas_x].select_dtypes(include=['float', 'int']).columns.size != len(columnas_x) or not df[columna_y].dtype.kind in 'i':
        resultado['Errores'].append("Los datos de entrada no cumplen con el tipo esperado (X como flotante o entero y Y como entero).")
        return df, None, resultado

    if df[columnas_x].isnull().any().any() or df[columna_y].isnull().any():
        resultado['Errores'].append("Hay valores nulos en las columnas seleccionadas.")
        return df, None, resultado

    # División de los datos en conjunto de entrenamiento y prueba
    X = df[columnas_x]
    y = df[columna_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    try:
        # Obtener las primeras 10 filas para las columnas especificadas
        columnas_unicas = list(set(columnas_x + [columna_y]))
        primeras_filas = df[columnas_unicas].head(10)
    

        # Entrenamiento del modelo
        modelo = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, max_iter=1000, random_state=1, early_stopping=True, validation_fraction=0.1)
        modelo.fit(X_train, y_train)
        
        # Predicción y cálculo del accuracy score
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        # Transformación del reporte de clasificación para la tabla
        reporte_clasificacion = []
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                reporte_clasificacion.append({
                    'clase': label,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1-score'],
                    'soporte': metrics['support']
                })
        
        # Gráfica de la Curva de Aprendizaje
        plt.figure(figsize=(10, 5))
        plt.plot(modelo.loss_curve_, label='Pérdida de Entrenamiento')
        if hasattr(modelo, 'validation_scores_'):
            plt.plot(modelo.validation_scores_, label='Precisión de Validación')
        plt.title('Curva de Pérdida y Precisión durante el Entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida / Precisión')
        plt.legend()
        unique_filename_loss = str(uuid.uuid4()) + '.png'
        loss_path = os.path.join(settings.TEMP_FILES_DIR, unique_filename_loss)
        plt.savefig(loss_path, dpi=400)
        plt.close()


        class_names = [str(x) for x in modelo.classes_] 

        
        # Creación del reporte
        reporte_data = {
            "titulo": "Reporte - Perceptrón Multicapa de Clasificación",
            "descripcion": f"Resultados de perceptrón multicapa utilizado para clasificar con variable objetivo (Y): [{columna_y}] y características (X): [{', '.join(columnas_x)}].",
            "tipo_analisis": "perceptron_multicapa",
            "resultados": {
                "nombres_columnas": columnas_unicas,
                "primeras_filas": primeras_filas.to_dict('records'),
                "classes": class_names,
                "matriz_confusion": conf_matrix.tolist(),
                "reporte_clasificacion": reporte_clasificacion,
                "path_imagen_perdida": unique_filename_loss,
                "estadisticas": {
                    "Accuracy": accuracy,
                    "hidden_layer_sizes": hidden_layer_sizes,
                    "activation": activation,
                    "solver": solver,
                    "alpha": alpha,
                    "Variable Objetivo": columna_y,
                    "Variables Predictoras": columnas_x
                }
            }
        }

        reporte_data['resultados']['matriz_confusion_indexada'] = list(enumerate(reporte_data['resultados']['matriz_confusion']))
        clases_y_filas = zip(reporte_data['resultados']['classes'], reporte_data['resultados']['matriz_confusion'])
        reporte_data['resultados']['clases_y_filas'] = list(clases_y_filas)

    except Exception as e:
        resultado['Errores'].append(str(e))
        reporte_data = None
    
    return df, reporte_data, resultado