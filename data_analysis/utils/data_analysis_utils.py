from utils.tablas_utils import crear_inicio_tabla

# Este script crea funciones para preparar datos para ser enviados a la vista, guardar resultados en la sesión y preparar datos para ser enviados por GET.
# Son funciones de apoyo para la vista de análisis de datos.
# Se utiliza en las vistas de data analysis y de reportes.
# Incluye las funciones guardar_resultado_en_sesion y preparar_datos_para_get.


def guardar_resultado_en_sesion(request, resultado):
    info_text = "\n".join(
        f"{key}: {', '.join(map(str, value)) if isinstance(value, list) else value}"
        for key, value in resultado.items() if value
    )
    request.session['info'] = info_text


def preparar_datos_para_get(df, include_dtypes=None, max_valores_unicos=20):
    df_html = crear_inicio_tabla(df)
    resultados = {}
    
    if include_dtypes:
        for dtype in include_dtypes:
            if dtype == 'category':
                # Filtra columnas que pandas reconoce como categóricas o que son 'object' con no más de max_valores_unicos valores únicos
                columnas = [col for col in df.columns if (df[col].dtype.name == 'category' or 
                                                         (df[col].dtype == 'object' and df[col].nunique() <= max_valores_unicos))]
            elif dtype == 'category_number':
                # Filtra columnas que son numéricas de tipo entero con no más de max_valores_unicos valores únicos
                columnas = [col for col in df.columns if (df[col].dtype == 'int64' and 
                                                         df[col].nunique() <= max_valores_unicos)]
            else:
                columnas = df.select_dtypes(include=[dtype]).columns.tolist()
            resultados[dtype] = columnas
    else:
        resultados['all'] = df.columns.tolist()
    
    # Si sólo hay un conjunto de columnas, se devuelve sólo ese conjunto junto con df_html
    if len(resultados) == 1:
        return df_html, list(resultados.values())[0]
    else:
        return df_html, resultados