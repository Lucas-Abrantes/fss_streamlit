import streamlit as st
import pandas as pd

st.title('FSS TELECOMUNICATIONS')

st.info('Aplicativo voltado para otimizações envolvendo superfícies seleticas de frequência.')

df_fss = pd.read_csv('https://raw.githubusercontent.com/Lucas-Abrantes/fss_streamlit/refs/heads/master/data.csv')
