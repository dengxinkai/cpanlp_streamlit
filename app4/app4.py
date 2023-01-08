import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="cpanlp的机器学习",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "很高兴您使用cpanlp的机器学习项目"
    }
)
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



