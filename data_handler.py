import re
import os
import pandas as pd
import numpy as np
from reddit_api import get_date, get_weekday
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import date, timedelta


def get_current_value():
    """Parsing current index value of S&P500 from yahoo finance
    
    Usage:     
    price_current = get_current_value()
    print(price_current)

    """
    url = "https://finance.yahoo.com/quote/%5EGSPC"
    html = urlopen(url)

    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text()

    m = re.search('\d,\d\d\d.\d\d', text)

    price_current_text = m[0].replace(',', '')
    return float(price_current_text)


def get_dataframe_all_subreddits():
    """Returns data frame of all available data
    
    Combines all csv files from each subreddit folder in the
    data directory to get a dataframe containing
    data of all subreddits

    """
    
    directory = r'data/'
    subreddits = [file for file in os.listdir(directory)]
    
    df_all_subreddits = pd.DataFrame({'title': [], 
                                      'score': [], 
                                      'id': [], 
                                      'url': [], 
                                      'comms_num': [], 
                                      'created': [], 
                                      'body': [],
                                      'timestamp': [],
                                      'subreddit': []})
    
    
    for subreddit in subreddits:
        data_dir = os.path.join('data', subreddit)
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        
        first = True
        for file in files:
            if first:
                df = pd.read_csv(file, index_col=0)
                first = False
            else: 
                df = pd.concat([df, pd.read_csv(file, index_col=0)], ignore_index=True)
        
        
        df = df.assign(subreddit = subreddit)
        
        df_all_subreddits = pd.concat([df_all_subreddits, df], ignore_index=True)
    
    return df_all_subreddits


def shift_timestamps_one_day(df):
    """
    Returns data frame with a shifted timestamp of one day
    """
    _timestamp = df["created"].apply(get_date)
    delta = timedelta(days=1)
    _timestamp += delta          
    df = df.assign(timestamp = _timestamp)
    df = df.rename(columns={"timestamp": "Date"})
       
    return df

def add_weekdays(df):
    """
    Returns data frame with a added column that is the weekday
    """
    _weekday = df["Date"].apply(get_weekday)
    df = df.assign(weekday = _weekday) 
    
    return df

def get_stock_data(ticker):
    """
    Returns data frame with a data of a stock from yahoo finance
    """
    df_stock = pd.read_csv(str(ticker) + '.csv')
    df_stock = df_stock.assign(DifferenceRelative = df_stock['Close'] / df_stock['Open'] - 1) 
    df_stock = df_stock.assign(Difference = df_stock['Close'] - df_stock['Open']) 
    df_stock['Target'] = np.where(df_stock['Difference'] > 0, 1, 0)
    df_stock['Date'] =  pd.to_datetime(df_stock['Date']).dt.date
    
    return df_stock

def get_data(start_date, end_date, subreddit):
     #Loading reddit data
    df_all_subreddits = get_dataframe_all_subreddits()
    df_all_subreddits = shift_timestamps_one_day(df_all_subreddits)
    df_all_subreddits = add_weekdays(df_all_subreddits)
    
    df_all_subreddits = df_all_subreddits[df_all_subreddits['Date'] >= start_date]
    df_all_subreddits = df_all_subreddits[df_all_subreddits['Date'] <= end_date]
    
    df_all_subreddits = df_all_subreddits[df_all_subreddits['subreddit'] == subreddit]
 
    
    #print(df_all_subreddits.shape)
    #print(df_all_subreddits.head())     
    #df_count_of_titles_per_day = df_all_subreddits.groupby(['timestamp']).count()['title']

    
    #Loading stock data
    ticker = '^GSPC'
    df_stock = get_stock_data(ticker)

    
    # Merging to data frames
    df_final = pd.merge(df_all_subreddits, df_stock, on='Date', how='inner')
    df_final = df_final[['title', 'Target', 'DifferenceRelative']]
    
    #As stocks are traded only monday - friday only these days should occour 
    #after the join:
    #print(df_final['weekday'].unique())
    
    
    return df_final


if __name__ == '__main__':
    df = get_data(start_date = date(2019, 1, 1), 
                  end_date =  date(2019, 1, 31), 
                  subreddit = 'worldnews')

    
