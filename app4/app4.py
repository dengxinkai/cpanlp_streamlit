import streamlit as st
import numpy as np
import pandas as pd
import base64
import seaborn as sns
import matplotlib.pyplot as plt

data = [(1, 2, 3)]
df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
uploaded_file = st.file_uploader("上传csv文件", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("csv导入成df成功")
    st.write(df)
    valx = st.sidebar.selectbox(
    '选择x变量',
    df.columns)
    valy = st.sidebar.selectbox('选择y变量',df.columns)
    if st.sidebar.button('Show Plots'):
        fig = plt.figure(figsize=(10, 4))
        plt.rcParams["font.sans-serif"]=["SimHei"]
        sns.scatterplot(x = valx, y = valy, data = df)
        st.pyplot(fig)
st.title('cpanlp自然语言处理项目')
st.header("Chart with two lines")


csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (右击保存为.csv的文件)'
st.markdown(href, unsafe_allow_html=True)
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='df.csv',
    mime='text/csv',
)
