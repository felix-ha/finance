"""
RNN Cell with sequence of word2vec vectors

"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import from_numpy, zeros

from neural_net_architectures import training_SGD_RNN, RNN

from validation import run_validation
from data_handler import get_imdb_data, get_word2vec



# Hyperparameter
seq_len = pad = 150
vocab_size = dummy_dim = 300
hidden_dim=100
n_layers=1
output_size=1

test_size = 0.1
random_state = 123
lr = 0.25
n_epochs = 20
batch_size = 25

n_bins_calibration = 10



# Data Handling

# Create basic data frame that contains text and target  
#df = get_imdb_data(data_dir=r'data\aclImdb', N_per_class=50)  
#df.to_pickle(r'temp\data.pkl')

# Start feature enineering
#df = pd.read_pickle(r'temp\data.pkl')

#One hot dummy vectors
#X, y = get_word2vec(df, vocab_size, seq_len)

#data_dict = {'X':X, 'y':y, 'vocab_size': vocab_size, 'seq_len': seq_len}

#with open(r'temp\Xy.pkl', 'wb') as pick:
#    pickle.dump(data_dict, pick)


with open(r'temp\Xy.pkl', 'rb') as pick:
    data_dict = pickle.load(pick)
    
X = data_dict['X']        
y = data_dict['y']
vocab_size = data_dict['vocab_size']
seq_len = data_dict['seq_len']



# Model

N, seq_len, dummy_dim = X.shape

input_size=dummy_dim


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

X_train_T = from_numpy(X_train).float()
y_train_T = from_numpy(y_train).float()  
X_val_T = from_numpy(X_val).float()
y_val_T = from_numpy(y_val).float()  

hidden = zeros(1, seq_len, hidden_dim).float()

model = RNN(input_size, seq_len, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)

training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
                 lr=lr, hidden_0=hidden, n_epochs=n_epochs, batch_size=batch_size)


model.eval()
y_prob_val = model.forward(X_val_T, hidden).view(-1).detach().numpy()
y_prob_train = model.forward(X_train_T, hidden).view(-1).detach().numpy()

df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})



result_dict = {'df_result_val': df_result_val, 'df_result_train': df_result_train,
               'training_losses':training_losses, 'valid_losses': valid_losses}


with open(r'temp\result.pkl', 'wb') as pick:
    pickle.dump(result_dict, pick)



#run_validation(n_bins = 10)