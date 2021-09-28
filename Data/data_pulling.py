# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 21:50:02 2021

@author: Robyn
"""

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import yaml
import io
with open(r'./Data/myapi.yml') as file:
    myapi=yaml.full_load(file)['myapi']  
    file.close()


def get_onetick_data(query,myapi):
    response = requests.get(query, auth=HTTPBasicAuth(myapi["ONETICK_USERNAME"], myapi["ONETICK_PASSWORD"])).content
  
    rawData = pd.read_csv(io.StringIO(response.decode('utf-8')))
    rawData.drop(index=rawData.index[rawData['#TIMESTAMP']=='#TIMESTAMP'],inplace=True)
    rawData.rename(columns={"#TIMESTAMP": "Time"}, inplace=True)
    rawData['Time'] = pd.to_datetime(rawData['Time'], unit='ms').dt.tz_localize('UTC')
    rawData['Time'] = rawData['Time'].dt.tz_convert('US/Eastern')
    rawData['Time'] = rawData['Time'].dt.tz_localize(None)
    rawData.index=rawData.Time
    rawData.drop(columns='Time',inplace=True)
    return rawData

"load and clean order book data"
def clean_order_book(order_book,time_slice=('09:40','15:50'),freq='100ms'):
    """clean the order book data pulled from one ticker:
        -remove data outside the time slice
        -resample to given fixed frequency
    note: resample will resulted in additional data on holiday, it won't change results, but need to removed for efficiency
    
    Parameters
    ----------
    order_book : TYPE
        oneTick order book data.
    time_slice : TYPE, optional
        only include data bewteen the time slice. The default is ('09:40','15:50').
    freq : TYPE, optional
        frequency of the output order book data. The default is '100ms'.

    Returns
    -------
    order_book_df : TYPE
        fixed frequency order book data.
        

    """

    order_book=order_book.between_time(time_slice[0],time_slice[1]).copy()
 
    order_book.columns=[col.lower() for col in order_book.columns]
    if 'symbol_name' in order_book.columns:
        order_book.drop(columns='symbol_name',inplace=True)
    if 'mid_quote' not in order_book.columns: 
        order_book['mid_quote']=(order_book.ask_price1+order_book.bid_price1)/2
    order_book=order_book.astype(float) 
    
    order_bin=order_book.resample(freq)
    
    order_book_df=order_bin.last()

    order_book_df=order_book_df.between_time(time_slice[0],time_slice[1]).copy()
    order_book_df.fillna(method='ffill',inplace=True)

    return order_book_df


# This only runs if called from command line
if __name__ == "__main__":
    ticker1='GOOG'
    ticker2='YELP'
    ticker3='QQQ'
    
    
    order_query1='https://data.onetick.com:443/omdwebapi/rest/?params={"context":"DEFAULT","query_type":"otq","otq":"124/720/otq/b17bd1a2-20ab-4859-8eaf-bc40d5f42a51.otq","enable_per_symbol_errors":"false","s":"20170824093000","e":"20170924160000","timezone":"America/New_York","response":"csv","format":["order=TIMESTAMP|BID_PRICE1|BID_PRICE2|BID_PRICE3|BID_PRICE4|BID_PRICE5|ASK_PRICE1|ASK_PRICE2|ASK_PRICE3|ASK_PRICE4|ASK_PRICE5|ASK_SIZE1|ASK_SIZE2|ASK_SIZE3|ASK_SIZE4|ASK_SIZE5|BID_SIZE1|BID_SIZE2|BID_SIZE3|BID_SIZE4|BID_SIZE5"]}'

    order_query2='https://data.onetick.com:443/omdwebapi/rest/?params={"context":"DEFAULT","query_type":"otq","otq":"124/720/otq/77613f64-6218-4322-9933-a471c711e0d0.otq","enable_per_symbol_errors":"false","s":"20170824093000","e":"20170924160000","timezone":"America/New_York","response":"csv","format":["order=TIMESTAMP|BID_PRICE1|BID_PRICE2|BID_PRICE3|BID_PRICE4|BID_PRICE5|ASK_PRICE1|ASK_PRICE2|ASK_PRICE3|ASK_PRICE4|ASK_PRICE5|ASK_SIZE1|ASK_SIZE2|ASK_SIZE3|ASK_SIZE4|ASK_SIZE5|BID_SIZE1|BID_SIZE2|BID_SIZE3|BID_SIZE4|BID_SIZE5"]}'
    order_query3='https://data.onetick.com:443/omdwebapi/rest/?params={"context":"DEFAULT","query_type":"otq","otq":"124/720/otq/4a595553-f127-4d86-b099-0360101f14a1.otq","enable_per_symbol_errors":"false","s":"20170824093000","e":"20170924160000","timezone":"America/New_York","response":"csv","format":["order=TIMESTAMP|BID_PRICE1|BID_PRICE2|BID_PRICE3|BID_PRICE4|BID_PRICE5|ASK_PRICE1|ASK_PRICE2|ASK_PRICE3|ASK_PRICE4|ASK_PRICE5|ASK_SIZE1|ASK_SIZE2|ASK_SIZE3|ASK_SIZE4|ASK_SIZE5|BID_SIZE1|BID_SIZE2|BID_SIZE3|BID_SIZE4|BID_SIZE5"]}'
    
    order_book1=get_onetick_data(order_query1,myapi)
    order_book2=get_onetick_data(order_query2,myapi)

    order_book3=get_onetick_data(order_query3,myapi)
    

    #%%
    order_book1.to_csv('%s_order_book.csv'%(ticker1))
    
    order_book2.to_csv('%s_order_book.csv'%(ticker2))
    
    
    order_book3.to_csv('%s_order_book.csv'%(ticker3))
