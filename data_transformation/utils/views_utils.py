from utils.tablas_utils import crear_inicio_tabla


def guardar_resultado_en_sesion(request, resultado):
    info_text = "\n".join(
        f"{key}: {', '.join(map(str, value)) if isinstance(value, list) else value}"
        for key, value in resultado.items() if value
    )
    request.session['info'] = info_text


def preparar_datos_para_get(df, include_dtypes=None, max_valores_unicos=20):
    df_html = crear_inicio_tabla(df)
    
    if include_dtypes:
        if 'category' in include_dtypes:
            # Filtra columnas que pandas reconoce como categóricas o que son 'object' con no más de 20 valores únicos
            columnas = [col for col in df.columns if (df[col].dtype.name == 'category' or 
                                                     (df[col].dtype == 'object' and df[col].nunique() <= max_valores_unicos))]
        else:
            if include_dtypes:
                columnas = df.select_dtypes(include=include_dtypes).columns.tolist()
            else:
                columnas = df.columns.tolist()
    return df_html, columnas