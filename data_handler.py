import re
import os
import glob
import pandas as pd
import numpy as np
from reddit_api import get_date, get_weekday
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import date, timedelta
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
from collections import Counter
from gensim.models import Word2Vec


def read_imdb_data(data_dir, N_per_class):
    """Loading imdb data files"""
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            counter = 0
            for f in files:
                counter += 1

                if counter > N_per_class:
                    break

                with open(f, encoding='utf8') as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)

            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)

    return data, labels


def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""

    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    return data_train, data_test, labels_train, labels_test


def get_dataframe_all_subreddits():
    """Returns data frame of all available data
    
    Combines all csv files from each subreddit folder in the
    data directory to get a dataframe containing
    data of all subreddits

    """
    
    directory = r'data/reddit/'
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
        data_dir = os.path.join('data', 'reddit',  subreddit)
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
    path = os.path.join('data', 'stocks', ticker + '.csv')
    df_stock = pd.read_csv(path)
    df_stock = df_stock.assign(DifferenceRelative = df_stock['Close'] / df_stock['Open'] - 1) 
    df_stock = df_stock.assign(Difference = df_stock['Close'] - df_stock['Open']) 
    df_stock['Target'] = np.where(df_stock['Difference'] > 0, 1, 0)
    df_stock['Date'] =  pd.to_datetime(df_stock['Date']).dt.date
    
    return df_stock


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


def get_reddit_data(start_date, end_date, subreddit, ticker):
    """
    Returns data frame with text, target and difference column of each date
    that lies in the given range from the selected subreddit
    """
     #Loading reddit data
    df_all_subreddits = get_dataframe_all_subreddits()
    df_all_subreddits = shift_timestamps_one_day(df_all_subreddits)
    df_all_subreddits = add_weekdays(df_all_subreddits)
    
    df_all_subreddits = df_all_subreddits[df_all_subreddits['Date'] >= start_date]
    df_all_subreddits = df_all_subreddits[df_all_subreddits['Date'] <= end_date]
    
    if subreddit is not None: 
        df_all_subreddits = df_all_subreddits[df_all_subreddits['subreddit'] == subreddit]
 
    
    #print(df_all_subreddits.shape)
    #print(df_all_subreddits.head())     
    #df_count_of_titles_per_day = df_all_subreddits.groupby(['timestamp']).count()['title']

    
    #Loading stock data
    df_stock = get_stock_data(ticker)

    
    # Merging to data frames
    df_final = pd.merge(df_all_subreddits, df_stock, on='Date', how='inner')
    df_final = df_final[['title', 'Target', 'DifferenceRelative', 'Date']]
    
    
    
    #Collapse Titles into one observation
    df_collapsed_titles = pd.DataFrame().reindex_like(df_final).dropna(axis='rows')   
    
    for date_curr in df_final['Date'].unique():
        df_curr = df_final[df_final['Date'] == date_curr]
        titles = ' '.join(df_curr['title'].values)
        
        df_collapsed_titles = df_collapsed_titles.append({'title': titles, 
                                    'Target': df_curr['Target'].iloc[0],
                                    'DifferenceRelative': df_curr['DifferenceRelative'].iloc[0],
                                    'Date': df_curr['Date'].iloc[0]}, ignore_index=True)

    
   
    
    
    #As stocks are traded only monday - friday only these days should occour 
    #after the join:
    #print(df_final['weekday'].unique())
    
    
    return df_collapsed_titles.rename(columns={'title': 'text', 'Target': 'target'})


def get_imdb_data(data_dir, N_per_class):
    data, labels = read_imdb_data(data_dir, N_per_class)
    
    
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    
    X = np.concatenate((train_X, test_X))
    y = np.concatenate((train_y, test_y))
    
    return pd.DataFrame(data = {'text': X, 'target': y})



def get_bag_of_words(df, max_features, verbose=False):
    vect = CountVectorizer(max_features=max_features, stop_words="english")
    vect.fit(df['text'].values)
       
    bag_of_words = vect.transform(df['text'].values) 
    
    feature_names = vect.get_feature_names()
         
    X = bag_of_words
    y = df['target'].values
    
    
    if verbose:
        print("Vocabulary size: {}".format(len(vect.vocabulary_)))
        print("Vocabulary content:\n {}".format(vect.vocabulary_))
        print("bag_of_words: {}".format(repr(bag_of_words)))
        print("Dense representation of bag_of_words:\n{}".format(
        bag_of_words.toarray()))
        print("Number of features: {}".format(len(feature_names)))
        print("Features (first 20):\n{}".format(feature_names[0:20]))
             
    
    return X, y, feature_names


def sentence_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words


def build_dict(data, vocab_size):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    train_X_flat = [elem for list in data for elem in list]
    word_count = dict(Counter(train_X_flat)) 
    
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} 
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): 
        word_dict[word] = idx + 2                            
        
    return word_dict


def convert_and_pad(word_dict, sentence, pad):
    NOWORD = 0 
    INFREQ = 1
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)


def convert_and_pad_data(word_dict, data, pad):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)


def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))            
    
    return one_hot

def get_one_hot_dummies(df, vocab_size, seq_len): 
    _title_words = df['text'].apply(sentence_to_words)
    df = df.assign(words = _title_words)
   
    word_dict = build_dict(df['words'].values, vocab_size)
      

    X, X_len = convert_and_pad_data(word_dict, df['words'].values, seq_len)
    y = df['target'].values

    X = one_hot_encode(X, n_labels=vocab_size)
    
    return X, y


def pad_sentence_of_words(sentence, pad):
    length = len(sentence)
    
    result = sentence
    
    if length > pad:
        result = sentence[0:pad]
        
    if length < pad:
        result[length-1:pad] = [' ' for i in range(pad- (length-1))]
       
       
    return result


def get_word_vectors(words, word2vec_model):
    return word2vec_model[words]


def get_word2vec(df, vocab_size, seq_len):
    _title_words = df['text'].apply(sentence_to_words)
    df = df.assign(words = _title_words)
    _words_padded = df['words'].apply(pad_sentence_of_words, pad=pad)
    df = df.assign(words_padded = _words_padded)
    
    # the empty string is used in padding if the sentence is too short
    # this string is converted with word2vec, maybe it is better to set the
    # word vector to zero. 
    word2vec_model = Word2Vec(df['words_padded'].values, size=dummy_dim, min_count=1)
  
    _word_vectors = df['words_padded'].apply(get_word_vectors, word2vec_model=word2vec_model)
    df = df.assign(word_vectors = _word_vectors)
         
    X = np.stack(df['word_vectors'].values)
    y = df['target'].values
    
    return X, y
    


if __name__ == '__main__':
    
    # Create basic data frame that contains text and target
    
    df = get_reddit_data(start_date = date(2018, 1, 1), 
                  end_date =  date(2018, 4, 28), 
                  subreddit = 'worldnews',
                  ticker = '^GSPC')
       
    
    #df = get_imdb_data(data_dir=r'data\aclImdb', N_per_class=50)  
    
    df.to_pickle(r'temp\data.pkl')
    
    
    # Start feature enineering
    
    df = pd.read_pickle(r'temp\data.pkl')
    
    
    # Word vectors 
    seq_len = pad = 100
    vocab_size = dummy_dim = 50 
    X, y = get_word2vec(df, vocab_size, seq_len)
    
    
    
    # Bag of words
    #X, y, feature_names = get_bag_of_words(df, max_features = 250)
    
    # One hot dummy vectors
    #seq_len = pad = 10
    #vocab_size = dummy_dim = 5
    #X, y = get_one_hot_dummies(df, vocab_size, seq_len)
    
    
    np.save(r'temp\X.npy', X)
    np.save(r'temp\y.npy', y)
    
    #X = np.load(r'temp\X.npy')
    #y = np.load(r'temp\y.npy')
    
    
    
    
    
    
    
    
    

    
