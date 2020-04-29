
from gensim.models import Word2Vec

import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from neural_net_architectures import training_full_batch_SGD_RNN, RNN
from torch import from_numpy, zeros, tensor

from validation import run_validation




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
    
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    
    train_X_flat = [elem for list in data for elem in list]
    word_count = dict(Counter(train_X_flat)) # A dict storing the words that appear in the reviews along with how often they occur
    
    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))            
    
    return one_hot


def convert_and_pad(word_dict, sentence, pad):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
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



def pad_sentence_of_words(sentence, pad):
    length = len(sentence)
    
    result = sentence
    
    if length > pad:
        result = sentence[0:pad]
        
    if length < pad:
        result[length-1:pad] = [' ' for i in range(pad- (length-1))]
       
       
    return result



# test implementation on imdb data set
df = pd.read_pickle('temp/imdb.pkl')

# use reddit data
#df = pd.read_pickle('temp/data.pkl')


seq_len = pad = 100
vocab_size = dummy_dim = 100

print('transform sentences to words')
#_title_words = df["title"].apply(sentence_to_words)
#df = df.assign(words = _title_words)
#df.to_pickle(r'temp\data.pkl')
df = pd.read_pickle('temp/data.pkl')


print('pad words')
_words_padded = df["words"].apply(pad_sentence_of_words, pad=pad)
df = df.assign(words_padded = _words_padded)


# the empty string is used in padding if the sentence is too short
# this string is converted with word2vec, maybe it is better to set the
# word vector to zero. 
word2vec_model = Word2Vec(df['words_padded'].values, size=dummy_dim, min_count=1)



def get_word_vectors(words):
    return word2vec_model[words]

_word_vectors = df['words_padded'].apply(get_word_vectors)
df = df.assign(word_vectors = _word_vectors)



X = np.stack(df['word_vectors'].values)
y = df['Target'].values



N, seq_len, dummy_dim = X.shape

input_size=dummy_dim
hidden_dim=120
n_layers=1
output_size=1


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=41, stratify=y)

X_train_T = from_numpy(X_train).float()
y_train_T = from_numpy(y_train).float()  
X_val_T = from_numpy(X_val).float()



hidden = zeros(1, seq_len, hidden_dim).float()


model = RNN(input_size, seq_len, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)

training_full_batch_SGD_RNN(model, X_train_T, y_train_T, lr=0.5, hidden_0=hidden, n_epochs=5)


model.eval()
y_prob_val = model.forward(X_val_T, hidden).view(-1).detach().numpy()
y_prob_train = model.forward(X_train_T, hidden).view(-1).detach().numpy()


df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})
df_result_val.to_pickle('temp/result_val.pkl')
df_result_train.to_pickle('temp/result_train.pkl')

run_validation(n_bins = 10)
















