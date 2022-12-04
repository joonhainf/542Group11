##LOGISTIC REGRESSION
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



#First split the data by a delimiter and store into variable data
df = pd.read_table("steam_games.csv", delimiter = ";", low_memory=False, nrows = 20000)

df['genre'] = df['Genre'].astype(str)
df['Free'] = [1 if 'Free to Play'  in x else 0 for x in df['genre']]

#Is CCU (Peak Number of Concurrent Players) related to the availability of discounts? 
X = df.iloc[:,[15]].values
y = df.iloc[:,23].values

#25% will be test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print(classification_report(y_test, y_pred))
plt.show()



