"""
Loading data from reddit: Saving a .csv file per day 
in the folder for the subreddit chosen
"""

import datetime
from datetime import date, timedelta
import calendar 
import time
import json
import requests
import praw
import pandas as pd
import time


def get_praw_instance():
    return praw.Reddit()

def get_pusshift_data(before, after):
    url = 'https://api.pushshift.io/reddit/submission/search/?after='+str(after)+'&before='+str(before) +'&sort_type=score&sort=desc&subreddit=worldnews'
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']
    
def get_date(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    return datetime.date(dt.year, dt.month, dt.day)

def get_timestamp(date):
    return time.mktime(date.timetuple())

def get_weekday(date):
    return calendar.day_name[date.weekday()]

def get_reddit_data(before, after, reddit):
    
    data  = get_pusshift_data(before, after)
    
    post_ids = [submission['id'] for submission in data]
    
    print('load '+ str(len(post_ids)) + ' posts beginning from date ' + str(get_date(after)))
    
    
    data = { "title":[], "score":[],
                    "id":[], "url":[], 
                    "comms_num": [], 
                    "created": [], 
                    "body":[]}
        
    for x in post_ids:
        submission = reddit.submission(id=x)
        
        data["title"].append(submission.title)
        data["score"].append(submission.score)
        data["id"].append(submission.id)
        data["url"].append(submission.url)
        data["comms_num"].append(submission.num_comments)
        data["created"].append(submission.created)
        data["body"].append(submission.selftext)
        
        
    return pd.DataFrame(data)




if __name__ == '__main__':
    reddit = get_praw_instance()
        
    
    start_date = date(2019, 5, 24)
    end_date = date(2020, 1, 1)
    delta = timedelta(days=1)
    
    
    
    first = True
    while start_date <= end_date:
        year = int(start_date.strftime("%Y"))
        day = int(start_date.strftime("%d"))
        month = int(start_date.strftime("%m"))
        after = int(get_timestamp(date(year, month, day)))  
        
        start_date += delta
        
        year = int(start_date.strftime("%Y"))
        day = int(start_date.strftime("%d"))
        month = int(start_date.strftime("%m"))
        before = int(get_timestamp(date(year, month, day)))
        
        try:
            df = get_reddit_data(before, after, reddit)
        except:
            print('retry... \n')
            time.sleep(60)
            df = get_reddit_data(before, after, reddit)
        
        df = get_reddit_data(before, after, reddit)
        df.to_csv(r'data/'+str(get_date(after))+'.csv')
