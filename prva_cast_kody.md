# ISI – Prvá časť predmetu: Kľúčové kódové vzory

Tento dokument sumarizuje **dôležité časti kódu / skelet** z cvičení – také, ktoré je typicky potrebné doplniť alebo vedieť napísať.

---

## 1. Základný workflow: tréning a testovanie modelu

### 1.1 Klasifikácia – Decision Tree + SVC na syntetických dátach

```python
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# generovanie dát
X, y = datasets.make_classification(n_samples=1000,
                                    n_features=3,
                                    n_redundant=0)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# rozhodovací strom
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Presnost klasifikacie dtc:", accuracy_score(y_pred, y_test))

# SVC s lineárnym jadrom
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Presnost klasifikacie svc:", accuracy_score(y_pred, y_test))
```

Dôležité bloky:

- `datasets.make_classification(...)`,
- `train_test_split(...)`,
- `model.fit(...)`,
- `model.predict(...)`,
- `accuracy_score(y_pred, y_test)`.

---

## 2. Decision Tree – vplyv `max_depth` na presnosť

```python
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n_features = 200
X, y = datasets.make_classification(750, n_features=n_features,
                                    n_informative=5, random_state=29)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

accuracies = []

# pozor: tu sa max_depth mení, nie počet featur
for d in np.arange(1, n_features+1, 2):
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    accuracies.append((preds == y_test).mean())

plt.plot(range(1, n_features+1, 2), accuracies, 'ko')
plt.title("Decision Tree Accuracy")
plt.ylabel("% Correct")
plt.xlabel("Max Depth")
plt.show()
```

Dôležitý vzor: cyklus cez rôzne hodnoty hyperparametra (`max_depth`), tréning stromu a ukazovanie presnosti.

---

## 3. Vizualizácia rozhodovacej hranice SVM v 2D

```python
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from itertools import product
from collections import namedtuple

X, y = datasets.make_blobs(n_features=2, centers=2)
svm = SVC(kernel='linear')
svm.fit(X, y)

Point = namedtuple('Point', ['x', 'y', 'outcome'])
decision_boundary = []

xmin, xmax = np.percentile(X[:, 0], [0, 100])
ymin, ymax = np.percentile(X[:, 1], [0, 100])

for xpt, ypt in product(np.linspace(xmin-2.5, xmax+2.5, 20),
                        np.linspace(ymin-2.5, ymax+2.5, 20)):
    p = Point(xpt, ypt,
              svm.predict(np.array([xpt, ypt]).reshape(1, -1)))
    decision_boundary.append(p)

f, ax = plt.subplots(figsize=(7, 5))
colors = np.array(['r', 'b'])

for xpt, ypt, pt in decision_boundary:
    ax.scatter(xpt, ypt, color=colors[pt[0]], alpha=.15)

ax.scatter(X[:, 0], X[:, 1], color=colors[y], s=30)
ax.set_ylim(ymin, ymax)
ax.set_xlim(xmin, xmax)
plt.show()
```

Kľúčové: vytvorenie gridu bodov, použitie `svm.predict` na každom bode, vykreslenie.

---

## 4. Funkcia na vykreslenie tvárí (Olivetti faces)

```python
import matplotlib.pyplot as plt

def print_faces(images, target, top_n):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        hspace=0.05, wspace=0.05)
    for i in range(top_n):
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        p.text(0, 14, str(target[i]))  # label triedy
        p.text(0, 60, str(i))          # index obrázku
```

Použitie: zobrazenie niekoľkých tvárí z datasetu Olivetti.

---

## 5. Random Forest – klasifikácia a pravdepodobnosti

### 5.1 Základná klasifikácia

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = datasets.make_classification(1000, random_state=500)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

rf = RandomForestClassifier(random_state=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:	", accuracy_score(y_test, y_pred))
print("Total Correct:	", (y_test == y_pred).sum())
```

### 5.2 Analýza `predict_proba` (kalibrácia)

```python
import pandas as pd
import matplotlib.pyplot as plt

probs = rf.predict_proba(X_test)
probs = np.around(probs, 1)
probs_df = pd.DataFrame(probs, columns=['0', '1'])
probs_df['was_correct'] = rf.predict(X_test) == y_test

f, ax = plt.subplots(figsize=(7, 5))
probs_df.groupby('0').was_correct.mean().plot(kind='bar', ax=ax)
ax.set_title("Accuracy at 0 class probability")
ax.set_ylabel("% Correct")
ax.set_xlabel("% trees for 0")
plt.show()
```

Dôležité: použitie `predict_proba`, vytvorenie DataFrame, groupby a vizualizácia.

---

## 6. Boston housing – načítanie dát a základná regresia

### 6.1 Načítanie Boston datasetu z URL

```python
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :],
                  raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
```

Výsledok:
- `data` – X (features),
- `target` – y (cena).

### 6.2 Scatter grafy (feature vs. target)

```python
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.25, random_state=33
)

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = y_train.min() - .5, y_train.max() + .5

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(12, 12)

for i in range(5):
    axes[i].set_title('Feature ' + str(i))
    axes[i].set_xlabel('Feature')
    axes[i].set_ylabel('Median house value')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, i], y_train)

plt.show()
```

### 6.3 SVR na Boston housingu + metriky

```python
from sklearn import svm, metrics

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.25, random_state=33
)

clf_svr = svm.SVR(kernel='linear')
clf_svr.fit(X_train, y_train)
print("Score on train ", clf_svr.score(X_train, y_train))

y_pred = clf_svr.predict(X_test)

print("Coefficient of determination: {0:.3f}".format(
    metrics.r2_score(y_test, y_pred)))
print("Mean absolute error: {0:.3f}".format(
    metrics.mean_absolute_error(y_test, y_pred)))
```

---

## 7. Titanic – imputácia a kódovanie kategórií

### 7.1 Imputácia veku a LabelEncoder

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

titanic_data = pd.read_csv('data/titanic.txt')
titanic_y = np.array(titanic_data.survived)

titanic_reduced = titanic_data.drop(columns=[
    "row.names", "survived", "name", "embarked",
    "home.dest", "room", "ticket", "boat"
])

# imputácia veku
ages = titanic_reduced.iloc[:, 1].values  # age column
mean_age = np.mean(ages[~np.isnan(ages)])
ages[np.isnan(ages)] = mean_age
titanic_reduced["age"] = ages

# kódovanie pohlavia
enc = LabelEncoder()
titanic_reduced["sex"] = enc.fit_transform(titanic_reduced["sex"])

print(titanic_reduced.loc[12])
```

### 7.2 One-hot encoding `pclass`

```python
from sklearn.preprocessing import OneHotEncoder

# label encoding pclass
enc = LabelEncoder()
titanic_reduced["pclass"] = enc.fit_transform(titanic_reduced["pclass"])

ohe = OneHotEncoder()
pclass_reshaped = np.array(titanic_reduced["pclass"]).reshape(-1, 1)
ohe_coded = ohe.fit_transform(pclass_reshaped).toarray()

# zahodenie pôvodného pclass a pridanie nových stĺpcov
titanic_reduced = titanic_reduced.drop(columns=["pclass"])
titanic_reduced["class 1"] = ohe_coded[:, 0]
titanic_reduced["class 2"] = ohe_coded[:, 1]
titanic_reduced["class 3"] = ohe_coded[:, 2]

print(titanic_reduced.loc[12])
```

---

## 8. Škálovanie – StandardScaler

```python
import numpy as np
from sklearn import preprocessing
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :],
                  raw_df.values[1::2, :2]])

X = data

print(X[:, :3].mean(axis=0))
print(X[:, :3].std(axis=0))

scaler = preprocessing.StandardScaler()
scaler.fit(X[:, :3])
X2 = scaler.transform(X[:, :3])

print("skalovane")
print(X2[:, :3].mean(axis=0))
print(X2[:, :3].std(axis=0))
```

---

## 9. Imputácia pomocou `SimpleImputer` na Iris

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data

# náhodné maskovanie
masking_array = np.random.binomial(1, .25, iris_X.shape).astype(bool)
iris_X[masking_array] = np.nan

print(iris_X[:5])

# mean strategy
imputer = SimpleImputer()
iris_X_prime = imputer.fit_transform(iris_X)
print("Imputovane")
print(iris_X_prime[:5])

# median strategy
imputer = SimpleImputer(strategy='median')
iris_X_prime = imputer.fit_transform(iris_X)
print("Imputovane median")
print(iris_X_prime[:5])
```

---

## 10. KFold a cross_val_score

### 10.1 Základný KFold na syntetických dátach

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold

X, y = make_regression(1001, shuffle=True, random_state=None)
kfold = KFold(n_splits=5)
output_string = "Fold: {}, N_train: {}, N_test: {}"

for i, (train, test) in enumerate(kfold.split(X)):
    print(output_string.format(i, len(y[train]), len(y[test])))
```

### 10.2 cross_val_score s RandomForestRegressor

```python
import numpy as np
from sklearn import datasets, ensemble, model_selection

X, y = datasets.make_regression(10000, 10, random_state=5)
rf = ensemble.RandomForestRegressor(max_features='log2')
scores = model_selection.cross_val_score(rf, X, y, cv=5)

print(scores)
print("Mean score: " + str(np.mean(scores)))
```

---

## 11. Ručný grid search pre DecisionTree + heatmapa

```python
from sklearn import datasets
import numpy as np
import itertools as it
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from matplotlib import cm

X, y = datasets.make_classification(n_samples=2000, n_features=10)
criteria = ['gini', 'entropy']
max_features = ['sqrt', 'log2', None]
parameter_space = it.product(criteria, max_features)

train_set = np.random.choice([True, False], size=len(y))
accuracies = {}

for criterion, max_feature in parameter_space:
    dt = DecisionTreeClassifier(criterion=criterion,
                                max_features=max_feature)
    dt.fit(X[train_set], y[train_set])
    accuracies[(criterion, max_feature)] = (
        dt.predict(X[~train_set]) == y[~train_set]
    ).mean()

cmap = cm.RdBu_r
f, ax = plt.subplots(figsize=(7, 4))
ax.set_xticks(np.arange(len(criteria)))
ax.set_yticks(np.arange(len(max_features)))
ax.set_xticklabels(criteria)
ax.set_yticklabels([str(f) for f in max_features])

plot_array = []
for max_feature in max_features:
    row = []
    for criterion in criteria:
        row.append(accuracies[(criterion, max_feature)])
    plot_array.append(row)

accuracy_values = list(accuracies.values())
vmin = np.min(accuracy_values) - 0.001
vmax = np.max(accuracy_values) + 0.001

colors = ax.matshow(plot_array, vmin=vmin, vmax=vmax, cmap=cmap)
f.colorbar(colors)
plt.show()
```

---

## 12. Pipeline + GridSearchCV – SVC (klasifikácia)

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# načítanie dát
X = np.load('X_data.npy', allow_pickle=True)
y = np.load('y_data.npy', allow_pickle=True)

pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC())])

param_range = [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = [
    {'clf__C': param_range,
     'clf__kernel': ['linear']},
    {'clf__C': param_range,
     'clf__gamma': param_range,
     'clf__kernel': ['rbf']},
    {'clf__C': param_range,
     'clf__gamma': param_range,
     'clf__degree': [1, 2, 3],
     'clf__kernel': ['poly']}
]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

gs.fit(X_train, y_train)

print('All scores: ', gs.cv_results_['mean_test_score'])
print('Best score: ', gs.best_score_)
print('Best parameters: ', gs.best_params_)
print('Test accuracy: ', gs.score(X_test, y_test))
```

---

## 13. Pipeline + SelectKBest + SVR + GridSearchCV – Boston (regresia)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :],
               raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_regression)),
    ('svr', SVR())
])

param_grid = {
    'select__k': [5, 8, 10, 12],
    'svr__kernel': ['linear', 'rbf'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto']
}

grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring='r2', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best parameters:", grid.best_params_)
print("Best CV R²:", grid.best_score_)
print("Test R²:", metrics.r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("SVR Prediction vs Actual (Boston Housing)")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.show()
```

---

## 14. Iris – Pipeline + multi-model GridSearchCV

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# poškodenie dát (NaN)
masking_array = np.random.binomial(1, 0.25, X.shape).astype(bool)
X[masking_array] = np.nan

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', MinMaxScaler()),
    ('clf', DecisionTreeClassifier())
])

param_grid = [
    {
        'imputer__strategy': ['mean', 'median'],
        'scaler': [MinMaxScaler(), StandardScaler(), None],
        'clf': [DecisionTreeClassifier()],
        'clf__criterion': ['gini', 'entropy', 'log_loss'],
        'clf__max_depth': [None, 3, 5, 8, 10, 15],
        'clf__min_samples_split': [2, 5, 8, 10, 15],
        'clf__min_samples_leaf': [1, 2, 4, 6, 8],
    },
    {
        'imputer__strategy': ['mean', 'median'],
        'scaler': [MinMaxScaler(), StandardScaler()],
        'clf': [LogisticRegression(max_iter=200)],
        'clf__C': [0.001, 0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__penalty': ['l2']
    }
]

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best model type:",
      type(best_model.named_steps['clf']).__name__)
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("Test accuracy:", accuracy_score(y_test, y_pred))
```

Kľúčový pattern:

- Pipeline s `imputer` + `scaler` + `clf`,
- `param_grid` ako **zoznam dictov** (viac modelov v jednom grid searchi),
- `GridSearchCV` robí hyperparameter tuning aj výber modelu naraz.
