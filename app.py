import streamlit as st

st.title('My first Streamlit app')

text = st.text_input('Enter some text:')
st.write('You entered:', text)

slider = st.slider('Select a range of values:', 0, 100)
st.write('You selected:', slider)