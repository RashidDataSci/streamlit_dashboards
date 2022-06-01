import streamlit as st
import seaborn as sns

st.header('This video is brought to you by #babaAammar')
st.text('Kia ap ko maza araha hy seekhnay main')

st.header('Pata nahi kia likhna hy?')

df = sns.load_dataset('iris')
st.write(df[['species','sepal_length','petal_length']].head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])