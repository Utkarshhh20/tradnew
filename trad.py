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
tickerSymbol='AAPL'
newscount=0
finviz_url = 'https://finviz.com/quote.ashx?t='
def news_headlines(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = bs(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    newstable = html.find(id='news-table')
    return newstable
	
# parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        st.write(x)
        # splite text in the td tag into a list 
        text = x.a.get_text()
	#vre
	date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])        
        # Set column names
        columns = ['date', 'time', 'headline']
        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)        
        # Create a pandas datetime object from the strings in 'date' and 'time' column
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
    st.write(parsed_news_df)    
    return parsed_news_df
st.subheader("Hourly and Daily Sentiment of {} Stock".format(tickerSymbol))
news_table = news_headlines(tickerSymbol)
typelab=type(news_table)
st.write(typelab)
parsed_news_df = parse_news(news_table)
#parsed_and_scored_news = score_news(parsed_news_df)
'''
def get_news():
    try:
        # Find news table
        news = pd.read_html(str(html), attrs = {'class': 'fullview-news-outer'})[0]
        st.write(news)
        news=news.drop(columns=[2])
        news=news.dropna()
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
'''
