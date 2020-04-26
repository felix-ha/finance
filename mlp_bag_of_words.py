import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from torch import from_numpy

from neural_net_architectures import FC_2_Layers_Binary_Output, training_full_batch_SGD
from validation import run_validation


max_features=100

df = pd.read_pickle('temp/data.pkl')

vect = CountVectorizer(max_features=max_features, stop_words="english")
vect.fit(df['title'].values)
bag_of_words = vect.transform(df['title'].values)

X = bag_of_words.toarray()
y = df['Target'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

X_train_T = from_numpy(X_train).float()
y_train_T = from_numpy(y_train).float()  
X_val_T = from_numpy(X_val).float()


model = FC_2_Layers_Binary_Output(max_features, 200, 75)

training_full_batch_SGD(model, X_train_T, y_train_T, lr = 0.01, n_epochs=2500)
    
    
model.eval()
y_prob_val = model.forward(X_val_T).detach().squeeze().numpy()
y_prob_train = model.forward(X_train_T).detach().squeeze().numpy()


df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})
df_result_val.to_pickle('temp/result_val.pkl')
df_result_train.to_pickle('temp/result_train.pkl')

run_validation(n_bins = 10)