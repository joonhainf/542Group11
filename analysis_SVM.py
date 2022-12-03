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
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from sklearn import svm, datasets


df = pd.read_table("steam_games.csv", delimiter = ";", low_memory=False, nrows = 1000)

##Create a binary variable which will be 0 if there is no discount and 1 if there is any discount

df['Has_Discount'] = ['Yes' if x==0 else 'No'  for x in df['Discount']]

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

##Linear SVM with the purpose of showing the support vectors vs the training data
classifier1 = SVC(kernel = 'linear', C=10, gamma=10)
classifier1.fit(X_train,Y_train)

support_vectors = classifier1.support_vectors_

plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('Negative Reviews')
plt.ylabel('Initial Price')
plt.show()