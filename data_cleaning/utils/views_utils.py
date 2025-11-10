from utils.tablas_utils import crear_inicio_tabla

# Funciones de apoyo para las vistas de limpieza de datos
# Se utilizan en las vistas de limpieza de datos
# Incluye las funciones guardar_resultado_en_sesion y preparar_datos_para_get

# Estas funciones son redundantes y deberían ser unificadas en un solo script junto con las de data_analysis para usarse globalmente
# se construyó así para simplificar la construccion y el entendimiento durante el desarrollo

def guardar_resultado_en_sesion(request, resultado):
    info_text = "\n".join(
        f"{key}: {', '.join(map(str, value)) if isinstance(value, list) else value}"
        for key, value in resultado.items() if value
    )
    request.session['info'] = info_text


def preparar_datos_para_get(df, include_dtypes=None):
    df_html = crear_inicio_tabla(df)
    if include_dtypes:
        columnas = df.select_dtypes(include=include_dtypes).columns.tolist()
    else:
        columnas = df.columns.tolist()
    return df_html, columnas