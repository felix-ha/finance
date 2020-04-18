import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


df = pd.read_pickle('temp/worldnews_2019.pkl')


accuracy_benchmark = df['Target'].agg(np.mean)



vect = CountVectorizer(max_features=50, stop_words="english")
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

max_leaf_nodes = 32
clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_leaf_nodes).fit(X, y)

y_pred = clf.predict(X)


accuracy = accuracy_score(y_true=y, y_pred=y_pred)

scores = cross_val_score(DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_leaf_nodes),
                         X, y, cv=10)


print("\nBenchmark accuray: {:.3f}".format(accuracy_benchmark))
print("Mean cv accuracy : {:.3f}".format(np.mean(scores)))
print("Train accuray    : {:.3f}\n".format(accuracy))


#df_features_importances = pd.DataFrame(dict(zip(feature_names, clf.feature_importances_)), index=[0])

df_features_importances = pd.DataFrame({'Feature': feature_names,
                                        'Importance': clf.feature_importances_})
df_features_importances = df_features_importances.sort_values(by=['Importance'], ascending=False)[0:15]
print(df_features_importances)
