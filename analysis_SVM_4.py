

#Luke Hwang's project "Group" 

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
from sklearn.preprocessing import StandardScaler
from sklearn import svm


#First split the data by a delimiter and store into variable data, nrows is 1000 for computational purposes.
df = pd.read_table("steam_games.csv", delimiter = ";", low_memory=False, nrows = 1900)

##Create a binary variable which will be 1 if the game is free to play.

df['tag'] = df['Tags'].astype(str)
df['Massive'] = [1 if 'FPS' in x and 'Multiplayer' in x else 0 for x in df['tag']]

pd.set_option('display.max_columns', 500)
df
#Correlation matrix
corr = df.corr()
print(corr)

##print(df['genre'].unique())
#df.sort_values(by=['Positive Reviews'], ascending=False)
#CCU and Positive Reviews seems to be most highly correlated with the presense of a discount.


training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)

test_size = len(test_set)

features = df[['Positive Reviews','Negative Reviews']]

label = df['Massive']

#Are CCU and Positive Reviews related to the availability of discounts? Here we set up the necessary training data for it.
X_train = training_set.iloc[:,[10,15]].values
Y_train = training_set.iloc[:,23].values
X_test = test_set.iloc[:,[10,15]].values
Y_test = test_set.iloc[:,23].values

#High C value of 10, such that the margin of the SVM is small and we are okay with misclassifying a few points.
classifier = SVC(kernel = 'rbf', C=1, gamma=10)
#This part will take a while
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred


#Utilize confusion matrix for accuracy computations.
cm = confusion_matrix(Y_test,Y_pred)
accuracy = cm.diagonal().sum()/len(Y_test)
print("\nSVM Accuracy : ", accuracy)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##Linear SVM with the purpose of showing the support vectors vs the training data
classifier1 = SVC(kernel = 'linear', C=10, gamma=10,probability=False)
#This part will take a while
classifier1.fit(X_train,Y_train)

support_vectors = classifier1.support_vectors_

plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('Positive Reviews')
plt.ylabel('CCU')
plt.show()
plt.clf()
plt.cla()
plt.close()

# Plotting the training set
fig, ax = plt.subplots(figsize=(12, 7))

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
# adding major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
ax.scatter(features[:-test_size]['Positive Reviews'], features[:-test_size]['Negative Reviews'], color="#8C7298")

#Logarithmic Scale for our dataset
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
plt.clf()
plt.cla()
plt.close()

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Removing to and right border
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_yscale('log')
ax.set_xscale('log')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
# Create grid to evaluate model
xx = np.linspace(-1, max(features['Positive Reviews']) + 1, len(X_train))
yy = np.linspace(0, max(features['Negative Reviews']) + 1, len(Y_train))
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
train_size = len(features[:-test_size]['Positive Reviews'])
# Assigning different colors to the classes
colors = Y_train
colors = np.where(colors == 1, '#8C7298', '#4786D1')
# Plot the dataset
ax.scatter(features[:-test_size]['Positive Reviews'], features[:-test_size]['Negative Reviews'], c=colors)
# Get the separating hyperplane
Z = model.decision_function(xy).reshape(XX.shape)
# Draw the decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# Highlight support vectors with a circle around them
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.ylabel('Negative Reviews')
plt.xlabel('Positive Reviews')
plt.show()
plt.clf()
plt.cla()
plt.close()

#There is not a clear way to split the data with a decision boundary, hence all amounts of positive reviews and ccus are relatively
#equally likely to have a great reputation.
#We are getting closer though!