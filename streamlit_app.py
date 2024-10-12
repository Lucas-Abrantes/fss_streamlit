import streamlit as st
import pandas as pd

st.title('FSS TELECOMUNICATIONS')

st.info('Aplicativo voltado para otimizações envolvendo superfícies seleticas de frequência.')

df_fss = pd.read_csv('https://raw.githubusercontent.com/Lucas-Abrantes/fss_streamlit/refs/heads/master/data.csv')

with st.expander('Data'):
  st.write('**Raw data')
  df_fss = pd.read_csv('https://raw.githubusercontent.com/Lucas-Abrantes/fss_streamlit/refs/heads/master/data.csv')
  df_fss

  st.write('**Inputs**'):
  X = df.drop(['BW1', 'BW2', 'RF1', 'RF2'], axis=1)   
  X

  st.write('**Outputs**'):
  y = df.drop(['h', 'p', 'd1', 'd2', 'w1', 'w2'], axis=1)
  y
