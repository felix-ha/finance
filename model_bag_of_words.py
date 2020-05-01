import numpy as np
import pandas as pd
import pickle


#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from validation import run_validation
from data_handler import get_bag_of_words, get_imdb_data



# Data Handling

# Create basic data frame that contains text and target
max_features = 250
#df = get_imdb_data(data_dir=r'data\aclImdb', N_per_class=1000)  
#df.to_pickle(r'temp\data.pkl')

# Start feature enineering
df = pd.read_pickle(r'temp\data_imdb_full.pkl')
# Bag of words
X, y, feature_names = get_bag_of_words(df, max_features = max_features)

# Save / load
data_dict = {'X':X, 'y':y, 'feature_names':feature_names}

with open(r'temp\Xy.pkl', 'wb') as pick:
    pickle.dump(data_dict, pick)


with open(r'temp\Xy.pkl', 'rb') as pick:
    data_dict = pickle.load(pick)
    
X = data_dict['X']        
y = data_dict['y']
feature_names = data_dict['feature_names']



# Model 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=185, stratify=y)


#max_leaf_nodes = 16
#model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_leaf_nodes).fit(X_train, y_train)

model = GradientBoostingClassifier(loss='deviance', learning_rate=0.05,
                                   n_estimators=100, subsample=0.8,
                                   criterion='friedman_mse', 
                                   min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_depth=3, 
                                   min_impurity_decrease=0.1, 
                                   min_impurity_split=None, 
                                   init=None, random_state=None, 
                                   max_features=None, verbose=0, 
                                   max_leaf_nodes=None, warm_start=False,
                                   presort='deprecated', validation_fraction=0.1,
                                   n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

model.fit(X_train, y_train)

y_prob_val = model.predict_proba(X_val)[:,1]
y_prob_train = model.predict_proba(X_train)[:,1]


df_result_val = pd.DataFrame(data = {'y_true': y_val, 'y_prob': y_prob_val})
df_result_train = pd.DataFrame(data = {'y_true': y_train, 'y_prob': y_prob_train})
df_result_val.to_pickle('temp/result_val.pkl')
df_result_train.to_pickle('temp/result_train.pkl')

run_validation(n_bins = 10)


df_features_importances = pd.DataFrame({'Feature': feature_names,
                                        'Importance': model.feature_importances_})
df_features_importances = df_features_importances.sort_values(by=['Importance'], ascending=False)[0:15]
print(df_features_importances)

