import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from data_handler import get_imdb_data, get_bag_of_words, get_tfidf
import pickle
from sklearn.model_selection import train_test_split
from neural_net_architectures import  training_SGD_Autoencoder
from validation import plot_losses, run_validation
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns


class AutoEncoder(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, code_size):
            super(AutoEncoder, self).__init__()
            self.input_size = input_size
            self.code_size = code_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.hidden_size_3 = hidden_size_3
            
            self.fc1_encoder = nn.Linear(self.input_size, self.hidden_size_1)
            self.fc2_encoder = nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.fc3_encoder = nn.Linear(self.hidden_size_2, self.hidden_size_3)
            self.fc4_encoder = nn.Linear(self.hidden_size_3, self.code_size)
            
            self.fc1_decoder = nn.Linear(self.code_size, self.hidden_size_3)
            self.fc2_decoder = nn.Linear(self.hidden_size_3, self.hidden_size_2)
            self.fc3_decoder = nn.Linear(self.hidden_size_2, self.hidden_size_1)
            self.fc4_decoder = nn.Linear(self.hidden_size_1, self.input_size)           
                        
            self.tanh = nn.Tanh()
            
        def encode(self, x):
            hidden1 = self.fc1_encoder(x)
            hidden1_tanh = self.tanh(hidden1)
            
            hidden2 = self.fc2_encoder(hidden1_tanh)
            hidden2_tanh = self.tanh(hidden2)
            
            hidden3 = self.fc3_encoder(hidden2_tanh)
            hidden3_tanh = self.tanh(hidden3)
            
            code = self.fc4_encoder(hidden3_tanh)
            code = self.tanh(code)
            
            return code
        
        def decode(self, code):
            
            hidden1 = self.fc1_decoder(code)
            hidden1_tanh = self.tanh(hidden1)
            
            hidden2 = self.fc2_decoder(hidden1_tanh)
            hidden2_tanh = self.tanh(hidden2)
            
            hidden3 = self.fc3_decoder(hidden2_tanh)
            hidden3_tanh = self.tanh(hidden3)
            
            output = self.fc4_decoder(hidden3_tanh)
            output = self.tanh(output)
            
            return output
            
            
            
        def forward(self, x):
            code = self.encode(x)
            output = self.decode(code)
           
            return output
        
max_features = dummy_dim = 250
 # Creating data set and features: Bag of words
# =============================================================================
# df = get_imdb_data(data_dir=r'data\aclImdb', N_per_class = None)  
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
              n_epochs=200, batch_size = 250, optimizer = optimizer, loss_func = loss_func)
    
    
model.eval()
with torch.no_grad():
    X_pred_val = model.forward(X_val_T)
    X_pred_train = model.forward(X_train_T)
    X_train_code = model.encode(X_train_T).numpy()
    X_val_code = model.encode(X_val_T).numpy()



#plot_losses(training_losses, valid_losses)



data = {'X1': X_train_code[:,0], 'X2': X_train_code[:,1], 'y': y_train}
df = pd.DataFrame(data)
ax = sns.scatterplot(x="X1", y="X2", hue="y", data=df)


data = {'X1': X_val_code[:,0], 'X2': X_val_code[:,1], 'y': y_val}
df = pd.DataFrame(data)
ax = sns.scatterplot(x="X1", y="X2", hue="y", data=df)




# Training on the encoded features:
# =============================================================================
# X_train = X_train_code
# X_val = X_val_code
# 
# model = GradientBoostingClassifier(loss='deviance', learning_rate=0.05,
#                                    n_estimators=100, subsample=0.8,
#                                    criterion='friedman_mse', 
#                                    min_samples_split=2, min_samples_leaf=1,
#                                    min_weight_fraction_leaf=0.0,
#                                    max_depth=3, 
#                                    min_impurity_decrease=0.1, 
#                                    min_impurity_split=None, 
#                                    init=None, random_state=None, 
#                                    max_features=None, verbose=0, 
#                                    max_leaf_nodes=None, warm_start=False,
#                                    presort='deprecated', validation_fraction=0.1,
#                                    n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
# 
# model.fit(X_train, y_train)
# 
# y_prob_val = model.predict_proba(X_val)[:,1]
# y_prob_train = model.predict_proba(X_train)[:,1]
# 
# 
# df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
# df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})
# df_result_val.to_pickle('temp/result_val.pkl')
# df_result_train.to_pickle('temp/result_train.pkl')
# 
# run_validation(n_bins = 10, n_bootstrap_samples=1000)
# 
# 
# df_features_importances = pd.DataFrame({'Feature': ['X1', 'X2'],
#                                         'Importance': model.feature_importances_})
# df_features_importances = df_features_importances.sort_values(by=['Importance'], ascending=False)[0:15]
# print(df_features_importances)
# =============================================================================
