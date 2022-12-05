#All the packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns




#First split the data by a delimiter and store into variable data
df = pd.read_table("steam_games.csv", delimiter = ";", low_memory=False)

##Create a binary variable which will be 0 if there is no discount and 1 if there is any discount
df['Has_Discount'] = [0 if x==0 else 1  for x in df['Discount']]

training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)

#Are Positive Reviews and CCU (Peak Number of Concurrent Players) related to the availability of discounts? Here we set up the necessary training data for it.
#Are Negative Reviews and Initial Price related to the availability of discounts? Here we set up the necessary training data for it.
X_train = training_set.iloc[:,[10,11,13,15]].values
y_train = training_set.iloc[:,22].values
X_test = test_set.iloc[:,[10,11,13,15]].values
y_test = test_set.iloc[:,22].values

scaler = StandardScaler()
# Fit only on X_train
scaler.fit(X_train)

# Scale both X_train and X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Run this
knn = KNeighborsClassifier(n_neighbors=7)
  
knn.fit(X_train, y_train)
  

print(knn.predict(X_test))


neighbors = np.arange(1, 9)
#Training and test accuracy for the loop later
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
# Loop over K values for i
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
      
    # Training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

y_pred = knn.best_estimator_.predict(X_test)
#Statistics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mean absolute error: {mae}')
print(f'mean squared error: {mse}')
print(f'rmse: {rmse}')
knn.score(X_test, y_test)

error = []

#Looping over things.$
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(y_test, pred_i)
    error.append(mae)


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='orange', 
         linestyle='dashed', marker='o',
         markerfacecolor='black', markersize=7)
         
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')
plt.show()


#Visualization
cm = confusion_matrix(y_test, y_pred) 

sns.heatmap(cm, annot=True, fmt='d')

print(classification_report(y_test, y_pred, zero_division = 1))
plt.show()

from sklearn.metrics import f1_score

f1s = []


for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    
    f1s.append(f1_score(y_test, pred_i, average='weighted'))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), f1s, color='orange', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=7)
plt.title('F1 Score K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.show()

#Classification Report
classifier4 = KNeighborsClassifier(n_neighbors=4)
classifier4.fit(X_train, y_train)
y_pred4 = classifier4.predict(X_test)
print(classification_report(y_test, y_pred4))