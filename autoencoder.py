import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from data_handler import get_imdb_data, get_bag_of_words
import pickle
from sklearn.model_selection import train_test_split
from neural_net_architectures import  training_SGD_Autoencoder
from validation import plot_losses


class AutoEncoder(nn.Module):
        def __init__(self, input_size, code_size):
            super(AutoEncoder, self).__init__()
            self.input_size = input_size
            self.code_size = code_size
            
            self.fc_encoder = nn.Linear(self.input_size,self.code_size)
            self.fc_decoder = nn.Linear(self.code_size, self.input_size)
            
            self.tanh = nn.Tanh()
            
         
         
            
        def forward(self, x):
            hidden1 = self.fc_encoder(x)
            code = self.tanh(hidden1)
            output = self.fc_decoder(code)
            

            return output
        
max_features = dummy_dim = 10  
 # Creating data set and features 
# =============================================================================
# df = get_imdb_data(data_dir=r'data\aclImdb', N_per_class=25)  
# df.to_pickle(r'temp\data.pkl')
# df = pd.read_pickle(r'temp\data.pkl')
# X, y, feature_names = get_bag_of_words(df, max_features = max_features)
# data_dict = {'X':X, 'y':y, 'feature_names':feature_names}
# with open(r'temp\Xy.pkl', 'wb') as pick:
#     pickle.dump(data_dict, pick)
# =============================================================================


# Loading data set
with open(r'temp\Xy.pkl', 'rb') as pick:
    data_dict = pickle.load(pick)
    
X = data_dict['X'].toarray()   
y = data_dict['y']     
feature_names = data_dict['feature_names']




model = AutoEncoder(input_size = dummy_dim, code_size=2)



train_size = 0.8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-train_size, random_state=1, stratify=y)

X_train_T = torch.from_numpy(X_train).float()
y_train_T = torch.from_numpy(y_train).float()  
X_val_T = torch.from_numpy(X_val).float()
y_val_T = torch.from_numpy(y_val).float()



lr = 0.1
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)   
training_losses, valid_losses = training_SGD_Autoencoder(model, X_train_T, y_train_T,  X_val_T, y_val_T, 
              n_epochs=20, batch_size = 40, optimizer = optimizer, loss_func = loss_func)
    
    
model.eval()
X_pred_val = model.forward(X_val_T)
X_pred_train = model.forward(X_train_T)



plot_losses(training_losses, valid_losses)


#df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
#df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})



#result_dict = {'df_result_val': df_result_val, 'df_result_train': df_result_train,
 #              'training_losses':training_losses, 'valid_losses': valid_losses}


#with open(r'temp\result.pkl', 'wb') as pick:
#    pickle.dump(result_dict, pick)



#run_validation(n_bins = 10, n_bootstrap_samples=100)

