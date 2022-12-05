#Needed packages
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
import seaborn as sns
import statsmodels.api as sm


df = pd.read_table("steam_games.csv", delimiter = ";", low_memory=False, nrows = 20000)

#Looking for Multiplayer FPS games
df['tag'] = df['Tags'].astype(str)
df['big_FPS'] = [1 if 'FPS' in x and 'Multiplayer' in x else 0 for x in df['tag']]

#Reshape the df for x for some reason
X = df['Positive Reviews'].values.reshape(-1,1)
y = df['big_FPS'].values


logr = linear_model.LogisticRegression()
logr.fit(X,y)

log_odds = logr.coef_
odds = np.exp(log_odds)

print(odds)

#Probabilities of games being a Multiplayer FPS. While cool, we do not know which games these are
def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(logit2prob(logr, X))

#New approach to get tangible results

print(df.describe())

x_train, x_test, y_train, y_test = train_test_split(df[['Positive Reviews','Negative Reviews']], df['big_FPS'], test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

#accuracy
score = logisticRegr.score(x_test, y_test)
print(score)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
predictions = logisticRegr.predict(x_test)

#Confusion matrix and heatmap, all that stuff
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual')
plt.xlabel('Predicted')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
print(classification_report(y_test, predictions))
plt.show()
