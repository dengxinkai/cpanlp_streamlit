import streamlit as st
import numpy as np

st.title('cpanlp自然语言处理项目')
st.header("Chart with two lines")
import matplotlib.pyplot as plt

f = plt.figure()
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20)

st.plotly_chart(f)
