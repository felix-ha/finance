import re
import os
import pandas as pd
from reddit_api import get_date, get_weekday
from urllib.request import urlopen
from bs4 import BeautifulSoup


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
    
    df_all_subreddits = pd.DataFrame(
    {'title': [], 'score': [], 'id': [], 'url': [], 'comms_num': [], 'created': [], 'body': [],
           'timestamp': [], 'weekday': [], 'subreddit': []})
    
    
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
        
            
            
        _timestamp = df["created"].apply(get_date)
        df = df.assign(timestamp = _timestamp)
        
        _weekday = df["timestamp"].apply(get_weekday)
        df = df.assign(weekday = _weekday)  
        
        df = df.assign(subreddit = subreddit)

        
        df_all_subreddits = pd.concat([df_all_subreddits, df], ignore_index=True)
    
    return df_all_subreddits

    
    



if __name__ == '__main__':
    df_all_subreddits = get_dataframe_all_subreddits()
    print(df_all_subreddits.shape)
    print(df_all_subreddits.head())     
    df_count_of_titles_per_day = df_all_subreddits.groupby(['timestamp']).count()['title']
    
    
