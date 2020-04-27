import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from validation import run_validation


df = pd.read_pickle('temp/data.pkl')

def get_bag_of_words(df, max_features):
    vect = CountVectorizer(max_features=max_features, stop_words="english")
    vect.fit(df['title'].values)
    
    
    print("Vocabulary size: {}".format(len(vect.vocabulary_)))
    #print("Vocabulary content:\n {}".format(vect.vocabulary_))
    
    
    bag_of_words = vect.transform(df['title'].values)
    print("bag_of_words: {}".format(repr(bag_of_words)))
    print("Dense representation of bag_of_words:\n{}".format(
    bag_of_words.toarray()))
    
    
    feature_names = vect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("Features (first 20):\n{}".format(feature_names[0:20]))
    
    
    
    X = bag_of_words
    y = df['Target'].values
    
    return X, y, feature_names


X, y, feature_names = get_bag_of_words(df, max_features = 10)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)


#max_leaf_nodes = 16
#model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_leaf_nodes).fit(X_train, y_train)

model = GradientBoostingClassifier(loss='deviance', learning_rate=0.03,
                                   n_estimators=150, subsample=0.8,
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




