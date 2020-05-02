"""
RNN Cell with sequence of word2vec vectors

"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement

import torch

from neural_net_architectures import training_SGD_RNN, RNN, LSTM, GRU, BiRNN, BiGRU, BiLSTM

from validation import run_validation
from data_handler import get_imdb_data, get_word2vec



# Hyperparameter
model_name = 'TestModel'
seq_len = pad = 142
vocab_size = dummy_dim = 300
hidden_dim_rnn=150
hidden_dim_fc = 50
drop_p=0.2
n_layers=1
output_size=1

test_size = 0.15
random_state = 12
lr = 0.15
n_epochs = 10
batch_size = 200

n_bins_calibration = 10



# Data Handling

# Start feature engineering
#df = pd.read_pickle(r'temp\data_imdb_full.pkl')
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

#ids = sample_without_replacement(n_population=len(y), n_samples=3000, random_state=5)
#X = X[ids]
#y = y[ids]

#data_dict = {'X':X, 'y':y, 'vocab_size': vocab_size, 'seq_len': seq_len}
#with open(r'temp\Xy.pkl', 'wb') as pick:
#    pickle.dump(data_dict, pick)



# Model

N, seq_len, dummy_dim = X.shape

input_size=dummy_dim


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

X_train_T = torch.from_numpy(X_train).float()
y_train_T = torch.from_numpy(y_train).float()  
X_val_T = torch.from_numpy(X_val).float()
y_val_T = torch.from_numpy(y_val).float() 

 

### RNN 

# =============================================================================
# hidden_0 = torch.zeros(1, seq_len, hidden_dim_rnn)
# 
# model = RNN(input_size, seq_len, output_size=output_size, hidden_dim_rnn=hidden_dim_rnn,
#             hidden_dim_fc=hidden_dim_fc, drop_p=drop_p, n_layers=n_layers)
# 
# training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
#                  lr=lr, hidden_0=hidden_0, n_epochs=n_epochs, batch_size=batch_size)
# =============================================================================

### BiRNN 

# =============================================================================
# hidden_0 = torch.zeros(2, seq_len, hidden_dim_rnn)
#  
# model = BiRNN(input_size, seq_len, output_size=output_size, hidden_dim_rnn=hidden_dim_rnn,
#             hidden_dim_fc=hidden_dim_fc, drop_p=drop_p, n_layers=n_layers)
# 
# training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
#                  lr=lr, hidden_0=hidden_0, n_epochs=n_epochs, batch_size=batch_size)
# =============================================================================


### GRU

# =============================================================================
# hidden_0 = torch.zeros(1, seq_len, hidden_dim_rnn)
# 
# model = GRU(input_size, seq_len, output_size=output_size, hidden_dim_rnn=hidden_dim_rnn,
#             hidden_dim_fc=hidden_dim_fc, drop_p=drop_p, n_layers=n_layers)
# 
# training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
#                  lr=lr, hidden_0=hidden_0, n_epochs=n_epochs, batch_size=batch_size)
# =============================================================================


### BiGRU

# =============================================================================
# hidden_0 = torch.zeros(2, seq_len, hidden_dim_rnn)
# 
# model = BiGRU(input_size, seq_len, output_size=output_size, hidden_dim_rnn=hidden_dim_rnn,
#             hidden_dim_fc=hidden_dim_fc, drop_p=drop_p, n_layers=n_layers)
# 
# training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
#                  lr=lr, hidden_0=hidden_0, n_epochs=n_epochs, batch_size=batch_size)
# =============================================================================


### LSTM 

# =============================================================================
# hidden_0 = torch.zeros(1, seq_len, hidden_dim_rnn)
# cell_0 = torch.zeros(1, seq_len, hidden_dim_rnn)
# hidden_0 = (hidden_0, cell_0)
# 
# model = LSTM(input_size, seq_len, output_size=output_size, hidden_dim_rnn=hidden_dim_rnn,
#             hidden_dim_fc=hidden_dim_fc, drop_p=drop_p, n_layers=n_layers)
# 
# training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
#                  lr=lr, hidden_0=hidden_0, n_epochs=n_epochs, batch_size=batch_size)
# =============================================================================


### BiLSTM 

hidden_0 = torch.zeros(2, seq_len, hidden_dim_rnn)
cell_0 = torch.zeros(2, seq_len, hidden_dim_rnn)
hidden_0 = (hidden_0, cell_0)

model = BiLSTM(input_size, seq_len, output_size=output_size, hidden_dim_rnn=hidden_dim_rnn,
            hidden_dim_fc=hidden_dim_fc, drop_p=drop_p, n_layers=n_layers)

training_losses, valid_losses = training_SGD_RNN(model, X_train_T, y_train_T, X_val_T, y_val_T,
                 lr=lr, hidden_0=hidden_0, n_epochs=n_epochs, batch_size=batch_size)





model.eval()
with torch.no_grad():
    y_prob_val = model.forward(X_val_T, hidden_0).view(-1).detach().numpy()
    y_prob_train = model.forward(X_train_T, hidden_0).view(-1).detach().numpy()

df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})



result_dict = {'df_result_val': df_result_val, 'df_result_train': df_result_train,
               'training_losses':training_losses, 'valid_losses': valid_losses,
               'model_name':model_name}


with open(r'temp\result.pkl', 'wb') as pick:
    pickle.dump(result_dict, pick)



run_validation(n_bins = n_bins_calibration, n_bootstrap_samples=500)
