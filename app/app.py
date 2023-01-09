import streamlit as st
import numpy as np
import pandas as pd
import base64
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
st.write("[返回](https://cpanlp.com/example/)")

data = [(1, 2, 3)]
df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
uploaded_file = st.file_uploader("上传csv文件", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("csv导入成df成功")
    st.write(df)

st.title('cpanlp自然语言处理项目')
st.header("Chart with two lines")


csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (右击保存为.csv的文件)'
st.markdown(href, unsafe_allow_html=True)

import matplotlib.pyplot as plt

f = plt.figure()
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20)

st.plotly_chart(f)
import yfinance as yf
import streamlit as st

st.write("""
# Simple Stock Price App
Shown are the stock **closing price** and ***volume*** of Google!
""")
 
# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbol = 'GOOGL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)
st.write("""
## Volume Price
""")
st.line_chart(tickerDf.Volume)
