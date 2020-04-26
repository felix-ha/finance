import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import from_numpy


from neural_net_architectures import FC_2_Layers_Binary_Output, training_full_batch_SGD
from validation import run_validation


def get_XOR_dataset(n_per_corner, sigma):
    X_00 = np.random.normal([0,0], sigma, size=(n_per_corner,2))
    y_00 = np.repeat(0, n_per_corner)
    X_11 = np.random.normal([1,1], sigma, size=(n_per_corner,2))
    y_11 = np.repeat(0, n_per_corner)
    
    X_01 = np.random.normal([0,1], sigma, size=(n_per_corner,2))
    y_01 = np.repeat(1, n_per_corner)
    X_10 = np.random.normal([1,0], sigma, size=(n_per_corner,2))
    y_10 = np.repeat(1, n_per_corner)
    
    X = np.concatenate((X_00, X_11, X_01, X_10))
    y= np.concatenate((y_00, y_11, y_01, y_10))
    return X, y





if __name__ == '__main__':
    X, y = get_XOR_dataset(n_per_corner=1000, sigma=0.01)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
    
    X_train_T = from_numpy(X_train).float()
    y_train_T = from_numpy(y_train).float()  
    X_val_T = from_numpy(X_val).float()
    
    
    model = FC_2_Layers_Binary_Output(2, 8, 8)
    
    training_full_batch_SGD(model, X_train_T, y_train_T, lr = 0.1, n_epochs=200)
        
        
    model.eval()
    y_prob_val = model.forward(X_val_T).detach().squeeze().numpy()
    y_prob_train = model.forward(X_train_T).detach().squeeze().numpy()
    
    
    df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
    df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})
    df_result_val.to_pickle('temp/result_val.pkl')
    df_result_train.to_pickle('temp/result_train.pkl')
    
    run_validation(n_bins = 10)
