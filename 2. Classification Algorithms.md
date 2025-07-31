## SUPERVISED LEARNING - Classification (Yes = 1 or No = 0)
[Link to full algorithm code on my personal Google Docs](https://docs.google.com/document/d/1y3ZkecbodvG_-noAnZCURoGXor-orNFvrV_73whGOWE/edit?usp=sharing)

In this document, I will be explaining on the second portion to supervised learning in machine learning, classification. I will also be providing the algorithms for it. The code here is generalised simply for easy adaptation to different datas sets.

- **Logistic Regression Model**
- **K-Nearest Neighbor (KNN) Model**
- **Decision Tree Classification Model**
- **Random Forest Classification Model**

---
## Logistic Regression

Goal : Determine whether 0 or 1, using 2 independent variables

Boundary between 0 and 1

Age, Estimated Salary to predict Purchased House (1) or Not (0)

Logistic Regression is based on probability (probability of a data pt being 0 or 1, based on features, Xi)

### Importing the libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### Importing the dataset

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  X = df.drop(y) # all rows, all feature columns → iloc [ : , 0:3]
y = dataset.iloc[:, 4].values        y= df[y]           # all rows, target column y only → iloc [ : , 4]
```

### Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # so data is not affected by scales (like how big, change it to how big in relation to one another from 0 to 1)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Fitting Logistic Regression to the Training set

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```

### Predicting the Test set results

```python
y_pred = classifier.predict(X_test)
```

### Making the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # checking for false positives etc, how many wrong predictions, right predictions
```

### Visualising the Training set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step =0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

### Visualising the Test set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step =
0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

As you notice in the visualization for the Test Set, most of the green dots fall under the green region (with a few red dots though because it’s hard to achieve 100% accuracy in logistic regression). This means our model could be good enough for predicting whether a person with a certain Age and Estimated Salary would purchase or not.

---

## K-Nearest Neighbors (K-NN)

Notice that Logistic Regression seems to have a linear boundary between 0s and 1s. As a result, it misses a few of the data points that should have been on the other side. 

KNN is a non-linear model that can capture these missed data points in a more accurate manner. KNN works by having a “new data point” and then counting how many neighbors belong to either category. 

If more neighbors belong to category A than category B, then the new point should belong to category A. Therefore, the classification of a certain point is based on the majority of its nearest neighbors (hence the name).

### Importing the libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### Importing the dataset

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  X = df.drop(y)
y = dataset.iloc[:, 4].values        y = df[y]
```

### Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Fitting K-NN to the Training set

```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p= 2)
classifier.fit(X_train, y_train)
```

### Predicting the Test set results

```python
y_pred = classifier.predict(X_test)
```

### Making the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

### Calculate the score (accuracy) on the test set

```python
accuracy = classifier.score(X_test, y_test)
print(f"KNN Classification Accuracy: {accuracy}")
```

```python
From sklearn.metrics import classification_report
classification_report(y_test,y_test) → precision, recall, f1 score, support
features = pd.Dataframe(classifier.feature_importances_, index = X.columns) → shows importance of each variable
```

### Visualising the Training set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step =0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

### Visualising the Test set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,
0].max() + 1, step = 0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array(\[X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y\_set)):
plt.scatter(X\_set\[y\_set == j, 0], X\_set\[y\_set == j, 1],
c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

```

## Decision Tree Classification

Creating a decision tree is about breaking down a dataset into smaller and smaller subsets while branching them out (creating an associated decision tree)

### Importing the libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### Importing the dataset

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values X = df.drop(y)
y = dataset.iloc[:, 4].values       y=df[y]
```

### Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Fitting Decision Tree Classification to the Training set

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

### Predicting the Test set results

```python
y_pred = classifier.predict(X_test)
```

### Making the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

### Calculate the score (accuracy) on the test set

```python
accuracy = classifier.score(X_test, y_test)
print(f"KNN Classification Accuracy: {accuracy}")
```

```python
From sklearn.metrics import classification_report
classification_report(y_test,y_test) → precision, recall, f1 score, support
features = pd.Dataframe(classifier.feature_importances_, index = X.columns) → shows importance of each variable
```

### Visualising the Training set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step =0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

### Visualising the Test set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,
0].max() + 1, step = 0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step =
0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

---

## Random Forest Classification

Recall from my previous post about Regression that a Random Forest is a collection of many decision trees. This also applies to Classification wherein many decision trees are used and the results are averaged.

### Importing the libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### Importing the dataset

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values X=df.drop(y)
y = dataset.iloc[:, 4].values       y=df[y]
```

### Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Fitting Random Forest Classification to the Training set

```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train) 
```

In a Random Forest model, n\_estimators refers to the number of individual decision trees that are built and combined to form the "forest."

### Predicting the Test set results

```python
y_pred = classifier.predict(X_test)
```

### Making the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

### Calculate the score (accuracy) on the test set

```python
accuracy = classifier.score(X_test, y_test)
print(f"KNN Classification Accuracy: {accuracy}")
```

```python
From sklearn.metrics import classification_report
classification_report(y_test,y_test) → precision, recall, f1 score, support
features = pd.Dataframe(classifier.feature_importances_, index = X.columns) → shows importance of each variable
```

### Visualising the Training set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step =
0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

### Visualising the Test set results

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step =
0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```



