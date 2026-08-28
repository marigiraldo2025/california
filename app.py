import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Título de la aplicación
st.title('Predicción del Precio Promedio de Casas en California')
st.write('Introduce los valores de las características para predecir el precio promedio de una casa.')

# Cargar el modelo de regresión lineal serializado
try:
    linear_model = joblib.load('linear_regression_model.joblib')
    st.success('Modelo de regresión lineal cargado exitosamente.')
except FileNotFoundError:
    st.error('Error: El archivo linear_regression_model.joblib no se encontró. Asegúrate de haberlo serializado.')
    st.stop()

# Cargar el escalador Min-Max serializado
try:
    scaler = joblib.load('minmax_scaler.joblib')
    st.success('MinMaxScaler cargado exitosamente.')
except FileNotFoundError:
    st.error('Error: El archivo minmax_scaler.joblib no se encontró. Asegúrate de haberlo serializado.')
    st.stop()

# Definir las características de entrada (de acuerdo con las columnas que usamos para entrenar X)
# Las características usadas fueron: MedInc, AveRooms, AveBedrms, AveOccup, Latitude
feature_names = ['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude']

# Crear campos de entrada para cada característica
input_data = {}
for feature in feature_names:
    # Usamos st.number_input para entradas numéricas
    # Se pueden agregar valores por defecto y rangos si se conocen
    # He puesto un valor por defecto y un rango tentativo basándome en los valores descriptivos del dataset.
    if feature == 'MedInc':
        input_data[feature] = st.number_input(f'MedInc (Ingreso medio del bloque en decenas de miles de dólares)', value=3.87, min_value=0.0, max_value=15.0)
    elif feature == 'AveRooms':
        input_data[feature] = st.number_input(f'AveRooms (Número promedio de habitaciones por hogar)', value=5.43, min_value=0.0, max_value=150.0)
    elif feature == 'AveBedrms':
        input_data[feature] = st.number_input(f'AveBedrms (Número promedio de dormitorios por hogar)', value=1.10, min_value=0.0, max_value=10.0)
    elif feature == 'AveOccup':
        input_data[feature] = st.number_input(f'AveOccup (Número promedio de ocupantes por hogar)', value=3.07, min_value=0.0, max_value=1250.0)
    elif feature == 'Latitude':
        input_data[feature] = st.number_input(f'Latitude (Latitud)', value=35.63, min_value=32.0, max_value=42.0)
    else:
        input_data[feature] = st.number_input(f'Introduce el valor para {feature}', value=0.0)

# Convertir los datos de entrada a un DataFrame de pandas
input_df = pd.DataFrame([input_data])

# Escalar las características de entrada usando el escalador cargado
# Asegúrate de que el orden de las columnas sea el mismo que el usado para el entrenamiento
input_scaled = scaler.transform(input_df[feature_names])

# Realizar la predicción cuando se presione un botón
if st.button('Predecir Precio'):
    prediction = linear_model.predict(input_scaled)
    # El target 'MedHouseVal' está en cientos de miles de dólares, así que multiplicamos por 100,000
    predicted_price = prediction[0] * 100000
    st.success(f'El precio promedio de la casa predicho es: ${predicted_price:,.2f}')
