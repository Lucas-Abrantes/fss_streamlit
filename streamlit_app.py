import streamlit as st
import pandas as pd

st.title('FSS TELECOMUNICATIONS')

st.info('Aplicativo voltado para otimizações envolvendo superfícies seletivas de frequência.')

# Carrega o dataset
df_fss = pd.read_csv('https://raw.githubusercontent.com/Lucas-Abrantes/fss_streamlit/refs/heads/master/data.csv')

with st.expander('Data'):
    st.write('**Raw data**')
    st.write(df_fss)

    st.write('**Inputs**')
    # Seleciona as colunas desejadas para X
    X = df_fss.drop(['BW1', 'BW2', 'RF1', 'RF2'], axis=1)   
    st.write(X)

    st.write('**Outputs**')
    # Seleciona as colunas desejadas para y
    y = df_fss.drop(['h', 'p', 'd1', 'd2', 'w1', 'w2'], axis=1)
    st.write(y)
