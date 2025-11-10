from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

def standardization(df, tipo, columnas_a_manipular):
    resultado = {
        'transformadas': [],
        'errores': [],
        'no_modificadas': []
    }

    if tipo not in ["minmax", "maxabs", "estandar"]:
        resultado['mensaje'] = "Método de estandarización inválido."
        return df, resultado
    if not columnas_a_manipular:
        resultado['mensaje'] = "No se especificaron columnas."
        return df, resultado

    scalers = {
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'estandar': StandardScaler()
    }

    # Verifica que las columnas existan y sean numéricas
    columnas_validas = df.select_dtypes(include=['number']).columns
    columnas_invalidas = [col for col in columnas_a_manipular if col not in columnas_validas]
    if columnas_invalidas:
        resultado['errores'].append(f"Columnas no válidas o no numéricas: {columnas_invalidas}")
        return df, resultado

    scaler = scalers[tipo]

    try:
        df[columnas_a_manipular] = scaler.fit_transform(df[columnas_a_manipular])
        resultado['transformadas'] = columnas_a_manipular
    except Exception as e:
        resultado['errores'].append(f"Error al aplicar el método {tipo}: {e}")

    return df, resultado