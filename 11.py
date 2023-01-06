import streamlit as st
st.title("cpanlp")
st.subheader('Number of pickups by hour')

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)
