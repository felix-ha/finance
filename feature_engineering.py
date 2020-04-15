"""
Combining all csv files from data directory to get a dataframe of all data
"""

import os
import pandas as pd
from reddit_api import get_date, get_weekday



data_dir = r'data/'
files = [os.path.join(data_dir, file) for file in os.listdir(r'data/')]

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

print(df.shape)
print(df.head())


df_count_of_titles_per_day = df.groupby(['timestamp']).count()['title']
