# Es crucial para el funcionamiento de la aplicación.
# Se utiliza en prácticamente todas las aplicaciones.
# Se encarga de limitar la cantidad de datos que se muestran en las tablas de la interfaz con el fin de no sobrecargar la página.
# El archivo csv_utils.py se encarga de continuar con la carga de las filas en caso de que se requiera.

def crear_inicio_tabla(df):
    return df.head(20).to_html(classes='table table', index=True)
