import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import requests

st.title('FSS TELECOMUNICATIONS')

st.info('Aplicativo voltado para otimizações envolvendo superfícies seletivas de frequência.')

# Carrega o dataset
df_fss = pd.read_csv('https://raw.githubusercontent.com/Lucas-Abrantes/fss_streamlit/refs/heads/master/data.csv')

with st.expander('Data'):
    st.write('**Raw data**')
    st.write(df_fss)

    st.write('**Inputs**')
    X = df_fss.drop(['BW1', 'BW2', 'RF1', 'RF2'], axis=1)   
    st.write(X)

    st.write('**Outputs**')
    y = df_fss.drop(['h', 'p', 'd1', 'd2', 'w1', 'w2'], axis=1)
    st.write(y)

# Baixar o modelo salvo
model_url = 'https://github.com/Lucas-Abrantes/fss_streamlit/blob/master/model.keras?raw=true'
model_path = 'model.keras'

# Baixar o arquivo do modelo
response = requests.get(model_url)
with open(model_path, 'wb') as f:
    f.write(response.content)

# Carregar o modelo salvo
model = load_model(model_path)

# Parâmetros ajustáveis pelo usuário
learning_rate = st.sidebar.slider('Learning Rate', 0.0001, 0.1, 0.009)
epochs = st.sidebar.slider('Epochs', 10, 500, 150)
batch_size = st.sidebar.slider('Batch Size', 16, 128, 32)
patience = st.sidebar.slider('Early Stopping Patience', 5, 50, 15)

# Atualizar a taxa de aprendizado do otimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

# Treinar o modelo
if st.button('Train Model'):
    with st.spinner('Training in progress...'):
        history = model.fit(
            input_train, output_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        # Avaliação do modelo
        y_pred = model.predict(input_test)
        mse = mean_squared_error(output_test, y_pred)
        r2 = r2_score(output_test, y_pred)
        mae = mean_absolute_error(output_test, y_pred)

        # Exibir as métricas
        st.write(f'MSE: {mse}')
        st.write(f'R2 Score: {r2}')
        st.write(f'MAE: {mae}')

        # Plotar a função de perda
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        st.pyplot(fig)

        # Salvar o histórico se necessário
        pd.DataFrame(history.history).to_csv('loss.csv', index=False)
