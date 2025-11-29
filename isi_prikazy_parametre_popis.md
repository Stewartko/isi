# Prehľad dôležitých príkazov, parametrov a jednoduchých popisov (Cvičenia 1–8)

Krátky „cheat sheet“ – najpoužívanejšie funkcie/triedy + základné parametre, ktoré sa objavili v kódoch.

---

## 1. `sklearn.datasets`

### `datasets.make_classification(...)`
Generuje syntetický dataset pre **klasifikáciu**.

Dôležité parametre:
- `n_samples` – počet vzoriek (riadkov).
- `n_features` – počet príznakov (stĺpcov).
- `n_informative` – počet „informatívnych“ príznakov (zvyšok môže byť šum).
- `n_redundant` – počet redundantných / lineárne závislých príznakov.
- `random_state` – seed generátora náhodných čísel (pre reprodukovateľnosť).

### `datasets.make_blobs(...)`
Generuje 2D/ND **zhluky** bodov (blobs), vhodné na vizualizáciu.

Dôležité parametre:
- `n_samples` – počet vzoriek.
- `n_features` – počet príznakov (často 2 kvôli grafu).
- `centers` – počet zhlukov (tried) alebo konkrétne súradnice centroidov.
- `cluster_std` – rozptyl okolo centroidu.
- `random_state` – seed.

### `datasets.make_regression(...)`
Generuje syntetické dáta pre **regresiu** (y je reálne číslo).

Dôležité parametre:
- `n_samples` – počet vzoriek.
- `n_features` – počet príznakov.
- `shuffle` – či premiešať poradie vzoriek.
- `random_state` – seed.

### `datasets.load_iris()`
Načíta vstavaný **Iris dataset** (3 triedy, 4 príznaky).  
Bez parametrov (prípadne `as_frame=True` v novších verziách).

---

## 2. `sklearn.model_selection`

### `train_test_split(X, y, test_size=..., random_state=...)`
Rozdelí dáta na **trénovaciu** a **testovaciu** množinu.

Parametre:
- `X` – vstupné dáta (matica vzoriek).
- `y` – ciele (labely).
- `test_size` – podiel dát pre test (napr. `0.25` = 25 %).
- `train_size` – alternatívne môžeš špecifikovať veľkosť train.
- `random_state` – seed, aby bol split vždy rovnaký.
- `shuffle` – či premiešať dáta (default True).

---

### `KFold(n_splits=...)`
Definuje **K-násobnú krížovú validáciu**.

Parametre:
- `n_splits` – počet foldov (napr. 5).
- `shuffle` – či sa pred delením miešajú indexy.
- `random_state` – seed pri shuffle.

Použitie:

```python
kfold = KFold(n_splits=5)
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
```

---

### `cross_val_score(estimator, X, y, cv=..., scoring=...)`
Vyhodnotí model pomocou krížovej validácie.

Parametre:
- `estimator` – model (napr. `RandomForestRegressor()`).
- `X`, `y` – dáta a ciele.
- `cv` – počet foldov alebo objekt (napr. `KFold`).
- `scoring` – metrika (napr. `"accuracy"`, `"r2"`).

Výstup: pole skóre pre jednotlivé foldy.

---

### `GridSearchCV(estimator, param_grid, cv=..., scoring=..., n_jobs=...)`
Vyhľadá najlepšiu kombináciu hyperparametrov pomocou grid search + CV.

Parametre:
- `estimator` – základný model alebo pipeline.
- `param_grid` – slovník alebo zoznam slovníkov: názov parametra → zoznam hodnôt.
  - pre pipeline: názov kroku a parametra, napr. `"clf__C"`.
- `cv` – počet foldov (napr. 5).
- `scoring` – metrika (napr. `"accuracy"`, `"r2"`).
- `n_jobs` – počet paralelných jobov (`-1` = všetky CPU).
- `verbose` – úroveň vypisovania priebehu.

Dôležité atribúty po fit:
- `.best_params_` – najlepšie parametre.
- `.best_score_` – najlepšie priemerné CV skóre.
- `.best_estimator_` – model s naj parametrami.

---

## 3. Klasifikátory a regresory (`sklearn.svm`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.linear_model`)

### `SVC(kernel=..., C=..., gamma=...)`
Support Vector Classifier (klasifikácia).

Parametre:
- `kernel` – typ jadra (`"linear"`, `"rbf"`, `"poly"`, `"sigmoid"`).
- `C` – regularizačný parameter (čím väčší, tým „tvrdšie“ rozhodnutie, menej regularizácie).
- `gamma` – šírka jadra pre `"rbf"` a `"poly"`:
  - `"scale"` – default, závisí od počtu features a var(X),
  - `"auto"` – 1 / n_features, alebo konkrétna hodnota (float).

---

### `SVR(kernel=..., C=..., gamma=...)`
Support Vector Regressor (regresia), parametre veľmi podobné ako SVC.

---

### `DecisionTreeClassifier(criterion=..., max_depth=..., max_features=...)`

Parametre:
- `criterion` – funkcia na meranie kvality splitu: `"gini"`, `"entropy"`, `"log_loss"`.
- `max_depth` – maximálna hĺbka stromu (None = bez limitu).
- `min_samples_split` – min. počet vzoriek na rozdelenie uzla.
- `min_samples_leaf` – min. počet vzoriek v listovom uzle.
- `max_features` – koľko príznakov skúšať pri splitte (`"sqrt"`, `"log2"`, int, float).

---

### `RandomForestClassifier(n_estimators=..., max_depth=..., random_state=...)`
Ansámbl (ensemble) viacerých rozhodovacích stromov pre klasifikáciu.

Parametre:
- `n_estimators` – počet stromov v lese.
- `criterion` – ako pri DecisionTreeClassifier.
- `max_depth` – max. hĺbka jednotlivých stromov.
- `max_features` – počet features skúšaných na split.
- `random_state` – seed (reprodukovateľnosť).
- `n_jobs` – paralelizácia.

---

### `RandomForestRegressor(...)`
Regresná verzia Random Forestu, parametre veľmi podobné ako `RandomForestClassifier`.

---

### `LogisticRegression(C=..., penalty=..., solver=..., max_iter=...)`
Lineárny klasifikátor s logistickou stratou.

Parametre:
- `C` – inverzná regularizácia (ako pri SVC, väčšie C = menej regularizácie).
- `penalty` – typ regularizácie (`"l2"`, `"l1"` pri vhodnom solveri).
- `solver` – optimalizačný algoritmus (`"liblinear"`, `"lbfgs"`, atď.).
- `max_iter` – max. počet iterácií pri učení.

---

## 4. Predspracovanie (`sklearn.preprocessing`, `sklearn.impute`, `sklearn.feature_selection`)

### `StandardScaler()`
Škálovanie na nulový priemer a jednotkovú smerodajku.

Metódy:
- `.fit(X)` – vypočíta priemer a std z dát.
- `.transform(X)` – transformuje X podľa uložených štatistík.
- `.fit_transform(X)` – fit + transform naraz.

### `MinMaxScaler()`
Škálovanie do intervalu [0, 1] (alebo iný interval).

Parametre:
- `feature_range` – dvojica (min, max), default (0, 1).

Metódy ako pri `StandardScaler`.

---

### `LabelEncoder()`
Kóduje kategórie do integerov (0, 1, 2, …).

Metódy:
- `.fit(y)` – uloží zoznam kategórií.
- `.transform(y)` – prevedie kategórie na integer kódy.
- `.fit_transform(y)` – fit + transform naraz.
- `.inverse_transform(y_int)` – vráti pôvodné kategórie.

Použitie v Titanic príklade: kódovanie `"male"/"female"` na 0/1.

---

### `OneHotEncoder()`
One-hot kódovanie kategórie do viacerých binárnych stĺpcov.

Dôležité parametre (novšie verzie):
- `sparse` / `sparse_output` – či výsledok bude sparse matica.
- `handle_unknown` – čo robiť s neznámymi kategóriami (`"error"`, `"ignore"`).

Metódy:
- `.fit(X_cat)` – naučí sa kategórie.
- `.transform(X_cat)` – vráti one-hot kódovanie.

---

### `SimpleImputer(strategy=...)`
Dopĺňa chýbajúce hodnoty (NaN).

Parametre:
- `strategy` – `"mean"`, `"median"`, `"most_frequent"`, `"constant"`.
- `fill_value` – hodnota pri `strategy='constant'`.

Metódy:
- `.fit(X)` – spočíta potrebné štatistiky (napr. priemer).
- `.transform(X)` – nahradí NaN hodnoty.
- `.fit_transform(X)` – fit + transform.

---

### `SelectKBest(score_func=..., k=...)`
Výber k najlepších príznakov podľa zvolenej scoring funkcie.

Parametre:
- `score_func` – funkcia na hodnotenie (napr. `f_regression` pre regresiu).
- `k` – počet vybraných features (napr. 5, 8, 10).

Metódy:
- `.fit(X, y)` – spočíta skóre pre každý feature.
- `.transform(X)` – vráti X s ponechanými len k najlepšími príznakmi.

---

## 5. `sklearn.pipeline.Pipeline`

### `Pipeline(steps=[(...), (...), ...])`
Poskladanie viacerých krokov (škálovanie, výber feature, model) do jedného objektu.

Parametre:
- `steps` – zoznam `(name, transformer/estimator)`:
  - napr. `[("scl", StandardScaler()), ("clf", SVC())]`.

Použitie:
- `.fit(X, y)` – postupne aplikuje `.fit` pre všetky kroky, posledný je model.
- `.predict(X)` – urobí `.transform` vo všetkých krokoch, potom `.predict` modelu.

Pri GridSearchCV:
- parametre jednotlivých krokov sa referencujú ako `"krok__parameter"`, napr. `"clf__C"`.

---

## 6. Metiky (`sklearn.metrics`)

### `accuracy_score(y_true, y_pred)`
Podiel správne klasifikovaných vzoriek.

### `r2_score(y_true, y_pred)`
R² pre regresiu (1 = dokonalé, <0 = horšie ako priemerný model).

### `mean_absolute_error(y_true, y_pred)`
Priemerná absolútna chyba (MAE).

---

## 7. Matplotlib – základné volania

### `plt.subplots(figsize=(w, h))`
Vytvorí figúru a os (figure + axes).  
- `figsize` – veľkosť v palcoch (šírka, výška).

Výstup:
- `fig` – objekt figúry,
- `ax` – objekt osi (na ňom voláš `scatter`, `plot`, `set_title`, ...).

### `ax.scatter(x, y, color=..., s=..., alpha=...)`
Bodový graf.
- `x`, `y` – súradnice bodov.
- `color` – farba bodov.
- `s` – veľkosť bodov.
- `alpha` – priehľadnosť.

### `ax.plot(x, y, ...)`
Čiarový graf.

### `plt.show()`
Zobrazí figúru.

---

## 8. Nízkoúrovňové utility

### `collections.namedtuple(name, fields)`
Vytvorí jednoduchý „immutable“ objekt (ako lightweight class) s danými poľami.

Použitie v 2D vizualizácii SVM:
- `Point = namedtuple("Point", ["x", "y", "outcome"])`

### `itertools.product(a, b, ...)`
Kartézsky produkt – všetky kombinácie prvkov z `a` a `b`.  
Použité na generovanie mriežky bodov v 2D (grid).

### `heapq.heappush`, `heapq.heappop`
Práca s min-heap (prioritný rad) pre Greedy/A*:
- `heappush(heap, (priority, additional_tie_break, state))`
- `heappop(heap)` – vráti prvok s **najmenším** `priority`.

### `time.time()`
Aktuálny čas v sekundách – rozdiel dvoch volaní = dĺžka behu algoritmu.

---

Tento prehľad pokrýva hlavné príkazy a parametre, ktoré sa objavili v kódoch od cvičenia 1 po Sudoku a 8-puzzle (hľadanie a heuristiky).
