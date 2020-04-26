import pandas as pd

import numpy as np
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

from sklearn.metrics import accuracy_score
from sklearn import metrics

def sentence_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words



pad = 500
vocab_size = 5000


df = pd.read_pickle('temp/data.pkl')
print('transform sentences to words')
_title_words = df["title"].apply(sentence_to_words)
df = df.assign(words = _title_words) 
df.to_pickle(r'temp\data.pkl')

df = pd.read_pickle('temp/data.pkl')





def build_dict(data, vocab_size = vocab_size):
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

print('build dictionary')
word_dict = build_dict(df["words"].values)


def convert_and_pad(word_dict, sentence, pad=pad):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=pad):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)


print('convert and padd')
X, X_len = convert_and_pad_data(word_dict, df["words"].values)
y = df['Target'].values


from sklearn.model_selection import train_test_split
X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)










import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        #x = x.t()
        #lengths = x[0,:]
        #reviews = x[1:,:]
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        
    
        out = out[:, pad-1]
        return self.sig(out.squeeze())



import torch
import torch.utils.data


# Turn the input pandas dataframe into tensors
train_sample_y = torch.from_numpy(y).float().squeeze()
train_sample_X = torch.from_numpy(X).long()

# Build the dataset
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build the dataloader
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)




def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model.forward(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
        
        


import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 100, vocab_size).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()
train(model, train_sample_dl, 120, optimizer, loss_fn, device)

model.eval()
y_prob_val = model.forward(torch.tensor(X_val).long()).detach().numpy()
y_prob_train = model.forward(torch.tensor(X).long()).detach().numpy()


df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y, 'y_prob': y_prob_train})
df_result_val.to_pickle('temp/result_val.pkl')
df_result_train.to_pickle('temp/result_train.pkl')






#test_review = 'Trump says China Hong Kong'
#test_data = sentence_to_words(test_review)
#test_data, test_data_len = convert_and_pad(word_dict, test_data, pad=pad)
#test_data = torch.tensor(np.array([test_data])).long()
#y_hat = model.forward(test_data)