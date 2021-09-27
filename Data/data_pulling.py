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
with open(r'myapi.yml') as file:
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

order_query1='https://data.onetick.com:443/omdwebapi/rest/?params={"context":"DEFAULT","query_type":"otq","otq":"124/720/otq/b17bd1a2-20ab-4859-8eaf-bc40d5f42a51.otq","enable_per_symbol_errors":"false","s":"20170824093000","e":"20170831160000","timezone":"America/New_York","response":"csv","format":["order=TIMESTAMP|BID_PRICE1|BID_PRICE2|BID_PRICE3|BID_PRICE4|BID_PRICE5|ASK_PRICE1|ASK_PRICE2|ASK_PRICE3|ASK_PRICE4|ASK_PRICE5|ASK_SIZE1|ASK_SIZE2|ASK_SIZE3|ASK_SIZE4|ASK_SIZE5|BID_SIZE1|BID_SIZE2|BID_SIZE3|BID_SIZE4|BID_SIZE5"]}'

ticker1='GOOG'

ticker2='YELP'
order_query2='https://data.onetick.com:443/omdwebapi/rest/?params={"context":"DEFAULT","query_type":"otq","otq":"124/720/otq/77613f64-6218-4322-9933-a471c711e0d0.otq","enable_per_symbol_errors":"false","s":"20170824093000","e":"20170831160000","timezone":"America/New_York","response":"csv","format":["order=TIMESTAMP|BID_PRICE1|BID_PRICE2|BID_PRICE3|BID_PRICE4|BID_PRICE5|ASK_PRICE1|ASK_PRICE2|ASK_PRICE3|ASK_PRICE4|ASK_PRICE5|ASK_SIZE1|ASK_SIZE2|ASK_SIZE3|ASK_SIZE4|ASK_SIZE5|BID_SIZE1|BID_SIZE2|BID_SIZE3|BID_SIZE4|BID_SIZE5"]}'

order_book1=get_onetick_data(order_query1,myapi)
order_book2=get_onetick_data(order_query2,myapi)


#%%
order_book1.to_csv('%s_order_book.csv'%(ticker1))

order_book2.to_csv('%s_order_book.csv'%(ticker2))


