#Luke Hwang's project "Group" 

#First split the data by a delimiter and store into variable data

#Can use SVM model to find distinctions between two genres of games.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import spacy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix



df = pd.read_table("steam_games.csv", delimiter = ";", low_memory=False, nrows = 10000)

##Create a binary variable which will be 0 if there is no discount and 1 if there is any discount

df['Has_Discount'] = [0 if x==0 else 1  for x in df['Discount']]

df.insert( "Has_Discount",df.["Discount"], True)

pd.set_option('display.max_columns', 500)
df
#Correlation matrix
corr = df.corr()
print(corr)
#Initial Price and Negative Reviews seems to be most highly correlated with the presense of a discount.


training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)

#Are Negative Reviews and Initial Price related to the availability of discounts?
X_train = training_set.iloc[:,[11,13]].values
Y_train = training_set.iloc[:,22].values
X_test = test_set.iloc[:,[11,13]].values
Y_test = test_set.iloc[:,22].values

#High C value of 10, such that the margin of the SVM is small and we are okay with misclassifying a few points.
classifier = SVC(kernel = 'rbf', C=10, gamma=10)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred

cm = confusion_matrix(Y_test,Y_pred)
accuracy = cm.diagonal().sum()/len(Y_test)
print("\nSVM Accuracy : ", accuracy)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.figure(figsize = (7,7))
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'orange'))(i), label = j)
plt.title('Apples Vs Oranges')
plt.xlabel('Weight In Grams')
plt.ylabel('Size in cm')
plt.legend()
plt.show()