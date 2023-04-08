# -*- coding: utf-8 -*-
"""web scraping.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BHtVeQzjT6G7cThP25_jN3_rzhIeTt2p
"""
import bs4

import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as ur

# Enter a stock symbol
index= 'MSFT'
# URL link 
url_is = 'https://www.sequoiacap.com/india/our-companies'
read_data = ur.urlopen(url_is).read() 
soup_is= BeautifulSoup(read_data,'lxml')

soup_is

ls= [] # Create empty list
for l in soup_is.find_all('div'): 
  #Find all data structure that is ‘div’
  ls.append(l.string) # add each element one by one to the list

ls

ls = [x for x in ls if x is not None]

ls

def newsletter():
  return ls

# Enter a stock symbol
index= 'MSFT'
# URL link 
url_is = 'https://seekingalpha.com/article/4577614-sequoia-fund-q4-2022-investor-letter'
read_data = ur.urlopen(url_is).read() 
soup_is1= BeautifulSoup(read_data,'lxml')

soup_is1

ls1= [] # Create empty list
for l in soup_is1.find_all('p'): 
  #Find all data structure that is ‘div’
  ls1.append(l.string)

ls1

def investmentletter():
  return ls1

