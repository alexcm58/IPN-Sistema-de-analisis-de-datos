import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

def apply_ordinal_encoder(df, column, order, resultado):
    if column not in df.columns:
        resultado['errores'].append(f"Columna '{column}' no encontrada.")
        return df
    if not isinstance(order, list) or not order:
        resultado['errores'].append("Se requiere un orden válido (lista no vacía) para aplicar el codificador ordinal.")
        return df
    if len(order) != len(set(order)):
        resultado['errores'].append("El orden contiene valores duplicados.")
        return df

    unique_values = set(df[column].dropna().unique())
    order_set = set(order)
    unknown_values = unique_values - order_set

    if unknown_values:
        resultado['errores'].append(f"Se encontraron categorías desconocidas {list(unknown_values)} en la columna '{column}' que no están en el orden proporcionado.")
        return df

    try:
        encoder = OrdinalEncoder(categories=[order])
        df[column] = encoder.fit_transform(df[[column]]).astype(int)
        resultado['resultados'].append(f"columna: {column}, tipo: codificador ordinal, orden: {order}")
    except Exception as e:
        resultado['errores'].append(f"Error al aplicar el codificador ordinal en {column}: {e}")
    return df

def apply_label_encoder(df, column, resultado):
    if column not in df.columns or df[column].isnull().any() or not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])):
        resultado['errores'].append(f"Columna '{column}' no es adecuada para el codificador de etiquetas o contiene valores nulos.")
        return df

    try:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        resultado['resultados'].append(f"columna: {column}, tipo: codificador de etiquetas")
    except Exception as e:
        resultado['errores'].append(f"Error al aplicar el codificador de etiquetas en {column}: {e}")
    return df

def apply_one_hot_encoder(df, column, resultado):
    if column not in df.columns or df[column].isnull().any() or not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])):
        resultado['errores'].append(f"Columna '{column}' no es adecuada para el codificador one-hot o contiene valores nulos.")
        return df

    try:
        encoder = OneHotEncoder(drop='first', dtype=int)
        encoded_cols = pd.DataFrame(encoder.fit_transform(df[[column]]).toarray(), columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df, encoded_cols], axis=1)
        df.drop(column, axis=1, inplace=True)
        resultado['resultados'].append(f"columna: {column}, tipo: codificador one-hot")
    except Exception as e:
        resultado['errores'].append(f"Error al aplicar el codificador one-hot en {column}: {e}")
    return df

def transformationStand(df, tipo, columnas_a_manipular, orders=None):
    resultado = {
        'resultados': [],
        'errores': []
    }

    transform_methods = {
        'labelencoder': apply_label_encoder,
        'one_hot': apply_one_hot_encoder,
        'ordinalencoder': apply_ordinal_encoder
    }

    # Verifica que ninguna columna a modificar contenga valores nulos
    columnas_con_nulos = [col for col in columnas_a_manipular if col in df.columns and df[col].isnull().any()]
    if columnas_con_nulos:
        resultado['errores'].append(f"Las siguientes columnas tienen valores nulos y no pueden ser procesadas: {', '.join(columnas_con_nulos)}.")
        return df, resultado

    if tipo not in transform_methods:
        resultado['errores'].append("Método de estandarización inválido.")
        return df, resultado

    if tipo == 'ordinalencoder' and not orders:
        resultado['errores'].append("Se requieren órdenes para la codificación ordinal.")
        return df, resultado
    
    for column in columnas_a_manipular:
        if column in df.columns:
            if tipo == 'ordinalencoder' and column in orders:
                df = transform_methods[tipo](df, column, orders[column], resultado)
            elif tipo != 'ordinalencoder':
                df = transform_methods[tipo](df, column, resultado)
            else:
                resultado['errores'].append(f"No se especificó orden para la columna '{column}'.")
        else:
            resultado['errores'].append(f"Columna '{column}' no encontrada en DataFrame.")

    return df, resultado
