# ISI – rýchly ťahák na zápočet

Minimalistický ťahák: veci, ktoré sa najčastejšie dajú čakať podľa cvičení + mailu.
(1) Python / sklearn vzory (1. časť), (2) algoritmy / heuristiky (2. časť).

---

## 1. Vzory kódu v sklearn (klasifikácia, regresia)

### 1.1 Základný pattern: dáta → split → model → metrika

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Presnosť:", acc)
```

- `test_size` – podiel dát na test (0.2–0.3).
- `random_state` – seed pre reprodukovateľnosť.
- `accuracy_score(y_true, y_pred)` – **poradie** je dôležité („najprv y_test“).

---

### 1.2 Generovanie a načítanie dát

```python
from sklearn import datasets

# klasifikácia
X, y = datasets.make_classification(
    n_samples=1000, n_features=3, n_redundant=0, random_state=29
)

# regresia
X, y = datasets.make_regression(1000, 10, random_state=5)

# 2D blobs (vizualizácia, SVM boundary)
X, y = datasets.make_blobs(n_features=2, centers=2)

# Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target
```

Titanic:

```python
import pandas as pd
import numpy as np

titanic_data = pd.read_csv("data/titanic.txt")
titanic_y = np.array(titanic_data.survived)

titanic_reduced = titanic_data.drop(
    columns=["row.names","survived","name","embarked",
             "home.dest","room","ticket","boat"]
)
```

Boston:

```python
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
```

---

### 1.3 Imputácia a škálovanie

#### SimpleImputer (chýbajúce hodnoty)

```python
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(strategy="mean")   # alebo "median"
X_imp = imputer.fit_transform(X)
```

Ručná imputácia veku (Titanic):

```python
ages = titanic_reduced.iloc[:, 1]   # stĺpec age (podľa poradia)
ages = np.array(ages)

mean_age = np.mean(ages[~np.isnan(ages)])
ages[np.isnan(ages)] = mean_age
titanic_reduced["age"] = ages
```

#### Škálovanie

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

minmax = MinMaxScaler()
X_mm = minmax.fit_transform(X)
```

- `StandardScaler` → priemer 0, smerodajná odchýlka 1.
- `MinMaxScaler` → interval [0,1].

---

### 1.4 Kódovanie kategórií – LabelEncoder, OneHotEncoder

```python
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
enc.fit(titanic_reduced["sex"])
titanic_reduced["sex"] = enc.transform(titanic_reduced["sex"])
```

Krátka verzia:

```python
titanic_reduced["sex"] = enc.fit_transform(titanic_reduced["sex"])
```

OneHotEncoder pre `pclass`:

```python
from sklearn.preprocessing import OneHotEncoder

label_encoder = enc.fit(titanic_reduced["pclass"])
titanic_reduced["pclass"] = label_encoder.transform(titanic_reduced["pclass"])

ohe = OneHotEncoder()
pclass_reshaped = np.array(titanic_reduced["pclass"]).reshape(-1, 1)
ohe_coded = ohe.fit_transform(pclass_reshaped).toarray()

titanic_reduced = titanic_reduced.drop(columns=["pclass"])
titanic_reduced["class 1"] = ohe_coded[:, 0]
titanic_reduced["class 2"] = ohe_coded[:, 1]
titanic_reduced["class 3"] = ohe_coded[:, 2]
```

- `reshape(-1, 1)` → spraví 2D stĺpcovú maticu z 1D vektora (nutné pre OHE).

---

### 1.5 Základné modely a ich parametre

#### DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(
    criterion="gini",        # alebo "entropy", "log_loss"
    max_depth=None,          # alebo int
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None        # "sqrt", "log2", int, float
)
```

#### RandomForestClassifier / Regressor

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=500
)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=0,
    max_features="log2"
)
```

`predict_proba` + jednoduchá kalibrácia:

```python
probs = rf_clf.predict_proba(X_test)
probs = np.around(probs, 1)
probs_df = pd.DataFrame(probs, columns=["0", "1"])
probs_df["was_correct"] = rf_clf.predict(X_test) == y_test
```

#### SVC / SVR

```python
from sklearn.svm import SVC, SVR

svc = SVC(kernel="linear", C=1.0, gamma="scale")
svc_rbf = SVC(kernel="rbf", C=1.0, gamma=0.1)
svr = SVR(kernel="rbf", C=1.0, gamma="scale")
```

- `C` – inverzná sila regularizácie (väčšie C = slabšia regularizácia).
- `gamma` – „dosah“ jadra (veľké gamma = úzke jadro, viac lokálne).

#### LogisticRegression

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=200
)
```

- `C` – sila regularizácie (väčšie C → menej regularizuje).
- `penalty='l2'` – L2 regularizácia.
- `solver='lbfgs'` – optimalizačný algoritmus.
- `max_iter` – max. počet iterácií solvera.

---

### 1.6 Cross-validation a GridSearchCV

#### KFold

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

#### cross_val_score

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator, X, y, cv=5)
print("Mean score:", scores.mean())
```

- `estimator` = ľubovoľný model (tree, RF, SVC, Pipeline...).

#### Pipeline + GridSearchCV (SVC)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([
    ("scl", StandardScaler()),
    ("clf", SVC())
])

param_range = [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = [
    {"clf__C": param_range, "clf__kernel": ["linear"]},
    {"clf__C": param_range,
     "clf__gamma": param_range,
     "clf__kernel": ["rbf"]},
    {"clf__C": param_range,
     "clf__gamma": param_range,
     "clf__degree": [1, 2, 3],
     "clf__kernel": ["poly"]}
]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring="accuracy",
                  cv=5,
                  n_jobs=-1)
gs.fit(X_train, y_train)
```

- mená krokov `("scl", ...)`, `("clf", ...)` sú ľubovoľné, ale v `param_grid` musíš použiť `scl__...`, `clf__...`.
- `best_params_`, `best_score_`, `best_estimator_` – výsledky grid searchu.

---

## 2. Algoritmy, heuristiky, vlastnosti (2. časť)

### 2.1 Definície

- **Kompletnosť** – algoritmus nájde riešenie vždy, keď existuje.
- **Optimálnosť** – algoritmus nájde riešenie s najnižšou možnou cenou.
- **Admisibilná heuristika** – nikdy nepreceňuje skutočnú cenu do cieľa:
  \( h(n) \leq h^*(n) \).
- **Konzistentná heuristika** – pre každú hranu platí:
  \( h(n) \leq c(n, n') + h(n') \).

---

### 2.2 BFS, DFS, Greedy, A* – stručná tabuľka

| Algoritmus       | OPEN štruktúra        | Kompletnosť                          | Optimálnosť (rovnaké náklady hrán)        |
|------------------|-----------------------|--------------------------------------|-------------------------------------------|
| **BFS**          | fronta (FIFO)         | áno (konečný graf)                  | áno                                       |
| **DFS**          | zásobník / rekurzia   | nie nutne (nekonečný graf)          | nie                                       |
| **Greedy best**  | min-heap podľa h(n)   | nie garantovaná                     | nie                                       |
| **A\***         | min-heap podľa g(n)+h(n) | áno (nezáporné hrany, admis. h) | áno (admisibilná + konzistentná h)        |

Kedy je A* **optimálny a kompletný**:
- náklady hrán \( c(n,n') \ge 0 \),
- heuristika je **admisibilná**, ideálne aj **konzistentná**.

---

### 2.3 Heuristiky pre 8-puzzle

- **Misplaced tiles**: počet nesprávne umiestnených dlaždíc.
- **Manhattan distance**: \( \sum |r - r_{goal}| + |c - c_{goal}| \) cez všetky dieliky.

Obe sú admisibilné; Manhattan je silnejšia a konzistentná → A* je s ňou optimálny.

---

### 2.4 CSP heuristiky (v Sudoku kóde)

- **MRV (Minimum Remaining Values)**  
  Vyber premennú (bunku), ktorej doména má **najmenej legálnych hodnôt**.

- **LCV (Least Constraining Value)**  
  Zorad hodnoty v doméne podľa toho, **ako málo obmedzujú** ostatné premenne (peers).

- **Forward Checking (FC)**  
  Po každom priradení:
  - pre každú ešte nepriradenú premennú prepočítaj doménu,
  - ak niektorá doména je prázdna → okamžitý backtrack (odrezanie vetvy).

Tieto nie sú súčasťou formálnej definície CSP, ale sú to **heuristiky a techniky pre efektívnejšie riešenie**.

---

### 2.5 Typické teoretické otázky (stručné odpovede)

- **Kedy je A* optimálny?**  
  Pri nezáporných nákladoch a **admisibilnej (ideálne konzistentnej) heuristike**.

- **Čo je admisibilná heuristika?**  
  Taká, ktorá nikdy nepreceňuje skutočnú cenu do cieľa (vždy \( h(n) \le h^*(n) \)).

- **Akú dátovú štruktúru používa BFS / DFS?**  
  BFS: fronta (FIFO), DFS: zásobník (LIFO) / rekurzia.

- **Je BFS optimálny?**  
  Áno, ak majú všetky hrany rovnaký náklad.

- **Je DFS optimálny a kompletný?**  
  Vo všeobecnosti nie: môže ísť „do nekonečna“ a nájsť neoptimálnu cestu.

- **Čo robí MRV / LCV / FC?**  
  MRV – výber premennej, LCV – výber hodnoty, FC – kontrola neprázdnych domén po priradení.

---

Toto je skrátený ťahák – mal by pokryť:
- najtypickejšie vzory kódu z 1. časti (sklearn),
- základné vlastnosti a heuristiky z 2. časti (BFS/DFS/Greedy/A*, MRV/LCV/FC).