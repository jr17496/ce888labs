from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.image as mpimg




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = "(%.2f)"%(cm[i, j])
        #print t
#         plt.text(j, i, t,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




df = pd.read_csv('bank-additional-full.csv',sep =';')
#print(df.head())

y_target = df[['y']].copy()
x_training = df.iloc[:,0:-1]

#print(y_target.head())
#print(x_training.head())

df_dummies = pd.get_dummies(df)
#print(df_dummies.head())

df_copy = df_dummies.drop(['y_no', 'duration'], axis=1).copy()
#print(df_copy.head())

y_histo = df_copy.iloc[:,-1]
x_training = df_copy.iloc[:,0:-1]

fig = plt.figure()
plt.hist(y_histo,  bins=2)
plt.plot()
fig.savefig('histogram.png')


clf = ExtraTreesClassifier(n_estimators = 100)
clf = clf.fit(x_training, y_histo)
scores = cross_val_score(clf, x_training, y_histo, cv=10, scoring = make_scorer(mse))
print("MSE: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std()))

y_pred = clf.predict(x_training)


confusion = confusion_matrix(y_histo, y_pred, labels=None, sample_weight=None)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion, classes=range(len(set(y_histo))), normalize = True,
                      title='Confusion matrix')

plt.savefig("confusion.png",bbox_inches='tight')


features = list(x_training)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(x_training.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]],  importances[indices[f]]))

# Plot the feature importances of the forest
fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(x_training.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_training.shape[1]), np.array(features)[indices])
plt.xlim([-1, x_training.shape[1]])
fig.set_size_inches(15,8)
axes = plt.gca()
axes.set_ylim([0,None])

plt.savefig("importances.png",bbox_inches='tight')











