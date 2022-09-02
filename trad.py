import streamlit as st
import hydralit_components as hc
import datetime
import pandas as pd 
import yfinance as yf
import numpy as np
import pandas_datareader as pdr
import mplfinance as fplt
import backtrader as bt 
import matplotlib.pyplot as plt
import talib
import matplotlib
import requests
import tweepy
import plotly.graph_objs as go
from rsi import RSIStrategy
from pandas.tseries.offsets import BDay
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
from string import Template
from datetime import date, timedelta
from yahoo_fin import stock_info as si 
from pandas_datareader import data as pdr
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import objective_functions
import plotly.express as px
import copy
from datetime import datetime
from io import BytesIO
from yahooquery import Screener
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bollingerband import BOLLStrat
from goldencrossover import goldencrossover
symbol='AAPL'
newscount=0
def get_news():
    try:
        # Find news table
        news = pd.read_html(str(html), attrs = {'class': 'fullview-news-outer'})[0]
        st.write(news)
        news=news.loc[0:1]
        st.write(news)
        links = []
        for a in html.find_all('a', class_="tab-link-news"):
            links.append(a['href'])
        
        # Clean up news dataframe
        news.columns = ['Date', 'News Headline']
        news['Article Link'] = links
        news = news.set_index('Date')
        return news

    except Exception as e:
        return e

url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()
html = bs(webpage, "html.parser")

news=get_news()
print(news)
for i in range(len(news)):
                    headline=news['News Headline'][i]
                    link=news['Article Link'][i]
                    st.write(f"{headline}: [More on this article]({link})")
                    newscount=newscount+1
                    if newscount<15:
                        st.write('____________________')
                    if newscount==15:
                        break
