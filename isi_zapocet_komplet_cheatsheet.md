# ISI – Zápočet: Kompletný prehľad (1. + 2. časť podľa mailu)

Toto je súhrnný „cheat sheet“ k zápočtu podľa:
- obsahu cvičení, ktoré si mi posielal,
- informácií z mailu (kompletnosť/optimalita algoritmov, širší záber otázok, pythonovské kódy).

---

## 0. Formát testu podľa mailu

- **10 otázok / 30 minút.**
- Otázky sú **sekvenčné** – *nedá sa vracať naspäť* na predchádzajúcu otázku.
- Z 1. časti predmetu (**machine learning**):
  - hlavne **Python/sklearn kódy**,
  - nielen z cvičení, ale môžu sa objaviť aj **kódy z domácich úloh a prednášok**.
- Z 2. časti:
  - otázky na **kompletnosť a optimalitu algoritmov**,
  - rozpoznanie algoritmov z *pseudokódu/kódu*,
  - čo používajú za dátové štruktúry, či môžu byť optimálne, či idú zapísať rekurzívne,
  - heuristiky (MRV, LCV, heuristiky pre 8‑puzzle) a forward checking.
- Nesprávna odpoveď = **0 bodov**, žiadne mínusové body.

---

## 1. Základy pojmov (ML časť)

### 1.1 Klasifikácia vs. regresia vs. zhlukovanie

- **Klasifikácia**
  - výstup: **kategória / trieda** (napr. 0/1, alebo 3 triedy Iris).
  - príklady modelov: `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `LogisticRegression`.

- **Regresia**
  - výstup: **reálne číslo** (napr. cena domu, hodnota y).
  - príklady modelov: `RandomForestRegressor`, `SVR`, syntetické dáta `make_regression`.

- **Zhlukovanie (clustering)**
  - **neučí sa zo značiek** (bez y), hľadá prirodzené zhluky v dátach.
  - v našich kódoch explicitne nebol algoritmus zhlukovania, ale `make_blobs` generuje dáta, ktoré *vyzerajú ako zhluky*.

---

## 2. Generovanie a načítanie dát (1. časť)

### 2.1 `sklearn.datasets` – syntetické dáta

#### `make_classification(...)` – syntetická klasifikácia

```python
from sklearn import datasets
X, y = datasets.make_classification(
    n_samples=1000,
    n_features=3,
    n_redundant=0,
    n_informative=5,   # ak používaš
    random_state=29    # alebo iné číslo
)
```

Dôležité parametre:
- `n_samples` – počet vzoriek (riadkov).
- `n_features` – počet feature stĺpcov.
- `n_informative` – počet skutočne „informatívnych“ príznakov.
- `n_redundant` – počet redundantných (lineárne kombinácie iných).
- `random_state` – seed pre reprodukovateľnosť.

#### `make_blobs(...)` – zhluky pre 2D vizualizáciu

```python
X, y = datasets.make_blobs(n_features=2, centers=2)
```

- `n_features=2` → môžeš kresliť v 2D.
- `centers=2` → dve triedy.

#### `make_regression(...)` – syntetická regresia

```python
X, y = datasets.make_regression(1000, 10, random_state=5)
```

- výstup `y` je reálna hodnota (regresia).

#### `load_iris()` – Iris dataset

```python
iris = datasets.load_iris()
X = iris.data
y = iris.target
```

---

### 2.2 Reálne dáta: Titanic & Boston

#### Titanic (`pandas.read_csv`)

```python
titanic_data = pd.read_csv('data/titanic.txt')
titanic_y = np.array(titanic_data.survived)
titanic_reduced = titanic_data.drop(
    columns=["row.names","survived","name","embarked",
             "home.dest","room", "ticket", "boat"]
)
```

- `drop(columns=...)` – vyhodíš nepotrebné stĺpce.

Imputácia veku ručne:

```python
ages = titanic_reduced.iloc[:, 1]  # stĺpec age
ages = np.array(ages)
mean_age = np.mean(ages[~np.isnan(ages)])
ages[np.isnan(ages)] = mean_age
titanic_reduced["age"] = ages
```

Kódovanie pohlavia:

```python
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
enc.fit(titanic_reduced["sex"])
titanic_reduced["sex"] = enc.transform(titanic_reduced["sex"])
```

One-hot encoding triedy lístka (pclass):

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

#### Boston Housing (regresia)

```python
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
```

- `data` – feature matica,
- `target` – cieľová premenná (mediánová cena domu).

---

## 3. Rozdelenie dát a krížová validácia

### 3.1 `train_test_split`

Typický riadok (vedieť doplniť):

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
```

### 3.2 `KFold` a ručné delenie

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### 3.3 `cross_val_score`

```python
from sklearn import ensemble
from sklearn import model_selection

rf = ensemble.RandomForestRegressor(max_features='log2')
scores = model_selection.cross_val_score(rf, X, y, cv=5)
print("Mean score:", np.mean(scores))
```

- vedieť: `cv=5` = 5-fold CV, `scores` je pole 5 hodnôt.

---

## 4. Predspracovanie dát

### 4.1 Imputácia chýbajúcich hodnôt – `SimpleImputer`

Iris príklad:

```python
from sklearn.impute import SimpleImputer
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data

masking_array = np.random.binomial(1, .25, iris_X.shape).astype(bool)
iris_X[masking_array] = np.nan

imputer = SimpleImputer(strategy='mean')
iris_X_mean = imputer.fit_transform(iris_X)

imputer = SimpleImputer(strategy='median')
iris_X_median = imputer.fit_transform(iris_X)
```

### 4.2 Škálovanie – `StandardScaler`, `MinMaxScaler`

Boston – škálovanie prvých troch feature:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled_first3 = scaler.fit_transform(X[:, :3])
```

Rozdiely:
- **StandardScaler**: presunie a škáluje tak, aby každý stĺpec mal **priemer 0** a **std 1**.
- **MinMaxScaler**: škáluje stĺpec na interval **[0, 1]** (alebo iný).

### 4.3 Kódovanie kategórií – `LabelEncoder`, `OneHotEncoder`

Už rozobraté pri Titanic – vedieť, že:
- LabelEncoder: kategórie → čísla (0,1,2,…),
- OneHotEncoder: kategórie → viac binárnych stĺpcov.

---

## 5. Modely (klasifikácia/regresia)

### 5.1 Decision Tree – `DecisionTreeClassifier`

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(
    criterion="gini",      # alebo "entropy", "log_loss"
    max_depth=5,           # None = bez limitu
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None      # môže byť "sqrt", "log2", int, float
)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
```

### 5.2 Random Forest – `RandomForestClassifier`, `RandomForestRegressor`

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=500
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

Dôležité:
- ansámbl stromov, hlasujú o výsledku,
- `predict_proba(X_test)` → pravdepodobnosti tried,
- interpretácia grafu „Accuracy at 0 class probability“ (kalibrácia).

### 5.3 SVM – `SVC`, `SVR`

Klasifikácia:

```python
from sklearn.svm import SVC

svc = SVC(kernel='linear', C=1.0, gamma='scale')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
```

Zmena jadra:

```python
SVC(kernel='rbf')
SVC(kernel='poly', degree=2)
```

Regresia (`SVR`) podobne, ale výsledkom sú reálne hodnoty.

### 5.4 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=200
)
log_reg.fit(X_train, y_train)
```

---

## 6. Model selection – Pipeline + GridSearchCV

### 6.1 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe_svc = Pipeline([
    ('scl', StandardScaler()),
    ('clf', SVC())
])
```

- `('scl', ...)` – meno kroku, používa sa v `param_grid` ako prefix: `scl__...`.
- `('clf', ...)` – klasifikátor, parametre `clf__C`, `clf__kernel`, …

### 6.2 GridSearchCV pre SVC

```python
from sklearn.model_selection import GridSearchCV

param_range = [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = [
    {'clf__C': param_range, 'clf__kernel': ['linear']},
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

gs.fit(X_train, y_train)
print('Best score:', gs.best_score_)
print('Best params:', gs.best_params_)
print('Test accuracy:', gs.score(X_test, y_test))
```

Vedieť:
- význam `param_grid` a zápisu `clf__C`, `clf__kernel`, …,
- čo je `.best_params_`, `.best_score_`, `.best_estimator_`,
- že `cv=5` znamená 5-násobnú krížovú validáciu.

### 6.3 Kombinovaný grid (DecisionTree + LogisticRegression)

Príklad z iris:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
```

Na teste môžu pýtať:
- doplniť jeden parameter,
- rozpoznať, že `clf` je raz tree, raz logistic regression,
- vysvetliť rozdiel medzi stratégiami imputácie/škálovania.

---

## 7. Metriky – `accuracy_score`, `r2_score`, `MAE`

```python
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

acc = accuracy_score(y_test, y_pred_cls)          # klasifikácia
r2 = r2_score(y_test, y_pred_reg)                 # regresia
mae = mean_absolute_error(y_test, y_pred_reg)     # regresia
```

RandomForest – graf „Accuracy at 0 class probability“:

```python
probs = rf.predict_proba(X_test)
probs = np.around(probs, 1)
probs_df = pd.DataFrame(probs, columns=['0', '1'])
probs_df['was_correct'] = rf.predict(X_test) == y_test

probs_df.groupby('0').was_correct.mean().plot(kind='bar')
```

- groupby podľa pravdepodobnosti triedy 0 (zaokrúhlenej),
- `was_correct.mean()` = empirická presnosť v danej skupine → kalibrácia pravdepodobností.

---

## 8. Algoritmy vyhľadávania – teória (2. časť)

### 8.1 Základné pojmy

- **Kompletnosť (completeness)**  
  Algoritmus je kompletný, ak **nájde riešenie vždy, keď existuje**.

- **Optimálnosť (optimality)**  
  Algoritmus je optimálny, ak **nájde riešenie s najnižšou možnou cenou** (napr. najkratšia cesta).

- **Admisibilná heuristika**  
  Heuristika `h(n)` je admisibilná, ak **nikdy nepreceňuje** skutočnú vzdialenosť do cieľa:
  \( h(n) \leq h^*(n) \) pre všetky n.

- **Konzistentná (monotónna) heuristika**  
  \( h(n) \leq c(n, n') + h(n') \) pre každú hranu (n → n').  
  Pri konzistentnej heuristike A* nemusí znovu otvárať uzly a je optimálny.

### 8.2 BFS, DFS, Greedy, A* – vlastnosti

| Algoritmus              | Open DS                  | Closed (visited) | Kompletnosť                         | Optimálnosť (unit cost)                     | Poznámka                                  |
|-------------------------|--------------------------|------------------|-------------------------------------|---------------------------------------------|-------------------------------------------|
| **BFS**                 | fronta (FIFO)            | áno              | áno (v konečnom grafe)             | áno, ak všetky hrany stoja rovnako          | veľké pamäťové nároky                    |
| **DFS (iter./rek.)**    | zásobník (LIFO) / stack  | áno/nie          | nie nutne (nekonečný graf)         | nie                                         | prirodzene rekurzívny                    |
| **Greedy best-first**   | min-heap podľa **h(n)**  | áno              | nie garantovaná                    | nie                                         | rýchly, ale môže zlyhať na lokálnych min |
| **A\***                | min-heap podľa **g+h**   | áno              | áno, pri h admisibilnej/konzistent.| áno, za tých istých podmienok               | kompromis medzi BFS a Greedy             |

#### BFS – pseudokód vzor

```text
Q := fronta so startom
visited := ∅

while Q nie je prázdna:
    s := Q.pop_front()
    ak s je cieľ:
        vráť riešenie
    pridaj s do visited
    pre každého nasledovníka n:
        ak n nie je vo visited ani v Q:
            Q.push_back(n)
```

#### DFS – pseudokód vzor (iteratívny)

```text
S := zásobník so startom
visited := ∅

while S nie je prázdny:
    s := S.pop()
    ak s je cieľ:
        vráť riešenie
    ak s nie je vo visited:
        pridaj s do visited
        pre každého nasledovníka n:
            S.push(n)
```

#### Greedy best-first – vzor

```text
OPEN := min-heap podľa h(n)
CLOSED := ∅

vloz start do OPEN

while OPEN nie je prázdne:
    n := vyber z OPEN s najmenším h(n)
    ak n je cieľ:
        vráť riešenie
    pridaj n do CLOSED
    pre každého nasledovníka s:
        ak s nie je v CLOSED:
            vypočítaj h(s)
            vlož s do OPEN
```

#### A* – vzor

```text
OPEN := min-heap podľa f(n) = g(n) + h(n)
CLOSED := ∅

g(start) := 0
f(start) := h(start)
vloz start do OPEN

while OPEN nie je prázdne:
    n := uzol s najmenším f(n) z OPEN
    ak n je cieľ:
        vráť riešenie
    presuň n z OPEN do CLOSED
    pre každého nasledovníka s:
        tentatívne_g := g(n) + cost(n, s)
        ak s v CLOSED a tentatívne_g >= g(s):
            pokračuj
        ak s nie je v OPEN alebo tentatívne_g < g(s):
            g(s) := tentatívne_g
            f(s) := g(s) + h(s)
            nastav parent(s) := n
            ak s nie je v OPEN:
                vlož s do OPEN
```

Na teste môžeš dostať kúsok takéhoto kódu a otázku: „Ktorý algoritmus je to?“ + doplniť riadok (napr. `f(s) = g(s) + h(s)`).

### 8.3 Heuristiky pre 8‑puzzle

- **Počet nesprávne umiestnených dlaždíc (misplaced tiles)**:
  - koľko dielikov je na zlej pozícii (0 ignoruješ).
  - jednoduchá, admisibilná, ale slabá.

- **Manhattanovská vzdialenosť**:
  - pre každý dielik spočítaš \\(|r - r_{cieľ}| + |c - c_{cieľ}|\\),
  - súčet pre všetky dieliky (okrem 0),
  - admisibilná a konzistentná → A* je s ňou optimálny.

### 8.4 CSP heuristiky: MRV, LCV, Forward Checking

Aj keď učiteľ píše, že „constraint section problémy nebudú“, MRV/LCV/FC boli priamo v Sudoku riešení – môžu sa objaviť **ako heuristiky v algoritme**, nie ako modelovanie CSP.

- **MRV (Minimum Remaining Values)**
  - pri výbere premennej zvolíš tú, ktorá má **najmenší počet legálnych hodnôt v doméne**,
  - „najviac obmedzená“ premenná.

- **LCV (Least Constraining Value)**
  - pre danú premennú zoradíš hodnoty podľa toho, **ako málo obmedzujú** ostatné premenné (peer bunky v riadku/stĺpci/bloku).

- **Forward Checking (FC)**
  - po priradení hodnoty skontroluješ pre každú ešte nepriradenú premennú, či doména nie je prázdna,
  - ak niektorá doména je prázdna → okamžitý backtrack (odrezanie vetvy).

Vedieť slovne opísať a rozpoznať kód:

```python
cell, domain = self.select_unassigned_mrv()
domain = self.sort_values_lcv(r, c, domain)

# forward checking
for rr in range(9):
    for cc in range(9):
        if self.board.grid[rr][cc] == 0:
            if len(self.board.get_domain(rr, cc)) == 0:
                all_domains_valid = False
```

---

## 9. Typy otázok, ktoré sa dajú čakať

### 9.1 „Doplň kód“ (1 riadok / parameter)

Príklady:
- `train_test_split` – doplniť `test_size` a `random_state`.
- `DecisionTreeClassifier(...)` – doplniť napr. `max_depth=...`.
- `RandomForestClassifier(...)` – doplniť `n_estimators=...`, `random_state=...`.
- `SimpleImputer(strategy='median')` – doplniť stratégiu.
- `StandardScaler()` – doplniť `fit_transform` na správne stĺpce.
- `GridSearchCV` – doplniť `param_grid` alebo `scoring='accuracy'` / `cv=5`.
- pipeline `param_grid` – doplniť `clf__C`, `clf__kernel`, atď.

### 9.2 „Identifikuj algoritmus z pseudokódu/kódu“

- BFS vs DFS vs Greedy vs A* (podľa použitej DS a f/h/g).
- Rozpoznať MRV/LCV/FC v Sudoku/8‑puzzle kóde.
- Rozpoznať, či ide o klasifikátor/regresor podľa použitého datasetu (`make_regression` vs `make_classification`).

### 9.3 „Teoretická vlastnosť“

- Je algoritmus **kompletný**? Za akých podmienok?
- Je **optimálny**? Pri akých nákladoch hrán?
- Akú dátovú štruktúru používa na otvorené stavy?
- Dá sa algoritmus implementovať rekurzívne? (DFS áno, BFS skôr nie).
- Čo je admisibilná heuristika? Ako súvisí s optimálnosťou A*?

---

Toto je maximum, čo sa dá odvodiť z:
- všetkých kódov, ktoré si posielal,
- zamerania cvičení,
- aj mailu od učiteľa (šírší záber, vlastnosti algoritmov, pythonovské kódy).
