import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from torch import from_numpy

from neural_net_architectures import FC_2_Layers_Binary_Output, training_SGD
from validation import run_validation
from data_handler import get_bag_of_words
import pickle

max_features = 250
#df = pd.read_pickle(r'temp\data_imdb_full.pkl')
#X, y, feature_names = get_bag_of_words(df, max_features = max_features)

#data_dict = {'X':X, 'y':y}

#with open(r'temp\Xy.pkl', 'wb') as pick:
#    pickle.dump(data_dict, pick)
    
with open(r'temp\Xy.pkl', 'rb') as pick:
    data_dict = pickle.load(pick)
    
X = data_dict['X'].toarray()        
y = data_dict['y']


train_size = 0.8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-train_size, random_state=1, stratify=y)
#X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1, stratify=y)

X_train_T = from_numpy(X_train).float()
y_train_T = from_numpy(y_train).float()  
X_val_T = from_numpy(X_val).float()
y_val_T = from_numpy(y_val).float()


model = FC_2_Layers_Binary_Output(max_features, 200, 75, 0.5)

training_losses, valid_losses = training_SGD(model, X_train_T, y_train_T,  X_val_T, y_val_T, 
             lr = 0.1, n_epochs=20, batch_size = 250)
    
    
model.eval()
y_prob_val = model.forward(X_val_T).detach().squeeze().numpy()
y_prob_train = model.forward(X_train_T).detach().squeeze().numpy()




df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})



result_dict = {'df_result_val': df_result_val, 'df_result_train': df_result_train,
               'training_losses':training_losses, 'valid_losses': valid_losses}


with open(r'temp\result.pkl', 'wb') as pick:
    pickle.dump(result_dict, pick)



run_validation(n_bins = 10, n_bootstrap_samples=100)