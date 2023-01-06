import streamlit as st
import numpy as np

st.title('cpanlp自然语言处理项目')
st.header("Chart with two lines")
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