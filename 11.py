import streamlit as st
st.title("cpanlp")
x = st.slider('Select a value')
st.write(x, 'squared is', x * x)
