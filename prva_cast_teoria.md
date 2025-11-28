# ISI – Prvá časť predmetu: Teoretický prehľad

Tento dokument sumarizuje **teóriu** z prvej časti predmetu podľa všetkých doteraz riešených cvičení.

---

## 1. Typy úloh: klasifikácia vs. regresia

### 1.1 Klasifikácia

- Cieľ: priradiť vstupnému vektoru \(x\) jednu z konečného počtu tried \(y \in \{0,1,\dots,K-1\}\).
- V cvičeniach:
  - Rozpoznávanie tvárí (Olivetti faces) – `SVC` (SVM klasifikátor).
  - Syntetické dáta z `make_classification` – porovnanie Decision Tree vs. SVC.
  - Random Forest Classifier na syntetických dátach.
  - Iris dataset – Decision Tree, Logistic Regression, multi-model `GridSearchCV`.
- Typická metrika: **accuracy** (presnosť)
  - \( \text{accuracy} = \frac{\#\text{správnych predikcií}}{\#\text{všetkých vzoriek}} \).

### 1.2 Regresia

- Cieľ: predikovať **spojitú** hodnotu \(y \in \mathbb{R}\).
- V cvičeniach:
  - `make_regression` – syntetické regresné dáta (RandomForestRegressor, KFold).
  - Boston housing dataset – predikcia ceny domu (`SVR`, `RandomForestRegressor`, `SVR` v pipeline).
- Typické metriky:
  - **R² (koeficient determinácie)**:
    - \( R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \),
    - 1 = ideál, 0 ≈ predikcia priemerom, < 0 = horšie než priemer.
  - **MAE (Mean Absolute Error)** – priemer absolútnych chýb.

---

## 2. Dátové sady použité v cvičeniach

### 2.1 Olivetti faces

- Dataset tvárí (64×64 grayscale obrázky).
- Cieľ: klasifikácia identity osoby.
- `fetch_olivetti_faces()`:
  - `faces.images` – 3D pole (n_samples × 64 × 64),
  - `faces.data` – 2D pole (n_samples × 4096),
  - `faces.target` – triedy (ID osoby).
- Vizualizácia mriežky tvárí pomocou vlastnej funkcie `print_faces`.

### 2.2 Syntetické dáta

- `datasets.make_classification` – binárna/multitriedna klasifikácia.
- `datasets.make_regression` – syntetická regresia.

Použitie:
- rýchle generovanie dát na testovanie modelov bez potreby reálnych datasetov.

### 2.3 Titanic (survival prediction)

- CSV súbor s informáciami o pasažieroch (vek, pohlavie, trieda lístka, atď.).
- Domáce úlohy a cvičenia:
  - imputácia veku,
  - kódovanie kategórií (`sex`, `pclass`),
  - one-hot encoding.

### 2.4 Boston housing

- Dataset s číselnými atribútmi (kriminalita, priemerný počet izieb, atď.) a targetom „medzná hodnota domu“.
- V cvičeniach:
  - ručné načítanie z `http://lib.stat.cmu.edu/datasets/boston`,
  - vizualizácia vzťahu feature → cena,
  - regresia (`SVR`, `RandomForestRegressor`),
  - feature selection (`SelectKBest`).

### 2.5 Iris

- 150 vzoriek, 4 featury (sepal/petal length/width), 3 triedy (druhy kvetu).
- Použité na:
  - imputáciu chýbajúcich hodnôt (pomocou `SimpleImputer`),
  - porovnanie Decision Tree vs Logistic Regression v jednom `GridSearchCV`,
  - demonštráciu pipeline (`imputer` → `scaler` → `clf`).

---

## 3. Základné modely a ich hyperparametre

### 3.1 Support Vector Machines – SVC a SVR

**SVC (Support Vector Classifier)**

- Klasifikačný model založený na nájdení optimálnej rozhodovacej hranice s maximálnou marginou.
- Kľúčové parametre:
  - `C` – penalizácia chybnej klasifikácie:
    - malé C → mäkšia margin, viac regularizácie,
    - veľké C → tvrdšia margin, menšia tolerancia chýb.
  - `kernel` – typ jadra:
    - `'linear'` – lineárna hranica,
    - `'rbf'` – RBF (Gaussian) jadro,
    - `'poly'` – polynomiálne jadro.
  - `gamma` (pre RBF/poly):
    - určuje „šírku“ jadra (lokálnosť rozhodnutí),
    - veľké `gamma` → veľmi lokálne hranice (riziko overfittingu).
  - `degree` – stupeň polynómu pri `kernel='poly'`.

**SVR (Support Vector Regression)**

- Regresná verzia SVM.
- Tie isté parametre (`C`, `kernel`, `gamma`, `degree`).
- Použitý v cvičeniach na Boston housingu (regresia ceny).

### 3.2 Rozhodovacie stromy (`DecisionTreeClassifier`)

- Hierarchický model, kde vnútorné uzly predstavujú testy na hodnotách featur, listy predstavujú triedy.
- Hyperparametre:
  - `criterion` – measure nečistoty:
    - `'gini'` – Giniho index,
    - `'entropy'` – entropia,
    - `'log_loss'` – logaritmická strata.
  - `max_depth` – maximálna hĺbka stromu,
  - `min_samples_split` – minimálny počet vzoriek potrebný na rozdelenie vnútorného uzla,
  - `min_samples_leaf` – minimálny počet vzoriek v listovom uzle,
  - `max_features` – počet featur, ktoré sa náhodne zvažujú pri každom splite (napr. `None`, `'sqrt'`, `'log2'`).

V cvičeniach:
- analýza accuracy vs. `max_depth`,
- ručný grid cez `criterion` × `max_features` a vizualizácia heatmapy.

### 3.3 Random Forest (klasifikácia/regresia)

- Ensemble rozhodovacích stromov:
  - každý strom trénovaný na bootstrap sample dát,
  - na každom node sa zvažuje náhodná podmnožina featur (controlled by `max_features`).
- Výhody:
  - znižuje varianciu, robustný voči overfittingu,
  - dobre funguje „out-of-the-box“.
- Kľúčové parametre:
  - `n_estimators` – počet stromov,
  - `max_features` – počet featur na každý split (`'sqrt'`, `'log2'`, číslo, podiel),
  - ostatné stromové parametre (`max_depth`, `min_samples_*`, …).

V cvičeniach:
- klasifikácia na `make_classification`,
- regresia na `make_regression`,
- analýza `predict_proba` a kalibrácie (accuracy vs. predikovaná pravdepodobnosť).

### 3.4 Logistic Regression

- Lineárny model pre klasifikáciu (binárnu aj multitrédu).
- Modeluje pravdepodobnosť tried cez logistickú/softmax funkciu.
- Hyperparametre:
  - `C` – inverzný koeficient regularizácie (L2),
  - `penalty` – typ regularizácie (`'l2'` v našich príkladoch),
  - `solver` – optimalizačný algoritmus (`'liblinear'`, `'lbfgs'`).
- V cvičeniach:
  - použitý v Iris pipeline ako alternatíva k Decision Tree.

---

## 4. Predspracovanie dát

### 4.1 Škálovanie

- **StandardScaler**
  - transformuje každý feature tak, že má priemer 0 a smerodajku 1.
  - vhodné pre modely citlivé na mierku (SVM, Logistic Regression, SVR).
- **MinMaxScaler**
  - škáluje do intervalu [0, 1].
  - zachováva monotónne vzťahy, ale mení rozsah.
- U stromových modelov (DecisionTree, RandomForest):
  - škálovanie nie je striktne potrebné, pretože rozdelenia závisia len od poradia hodnôt, nie od mierky.

### 4.2 Imputácia chýbajúcich hodnôt

- **SimpleImputer**
  - `strategy='mean'` – nahradenie chýbajúcich hodnôt priemerom,
  - `strategy='median'` – nahradenie mediánom,
  - (všeobecne aj `most_frequent`, `constant`).
- V cvičeniach:
  - Iris: umelo generované NaN v 25 % hodnôt,
  - Titanic: imputácia veku (pôvodne numpy/pandas, potom cez `SimpleImputer`).

### 4.3 Kódovanie kategórií

- **LabelEncoder**
  - premiena textové kategórie na integer kódy (napr. `male` → 1, `female` → 0),
  - v Titanic cvičení: kódovanie pohlavia.
- **OneHotEncoder**
  - premiena kategórie na binárne featury (one-hot vektor),
  - Titanic: `pclass` → `class 1`, `class 2`, `class 3`,
  - výhoda: lineárne modely neinterpretujú integer kódy ako „poradie“.


### 4.4 Feature selection

- **SelectKBest(score_func=f_regression)**:
  - univariačný výber K najlepších featur podľa F-testu (pre regresiu).
  - Použitý v pipeline s `SVR` nad Boston housing:
    - `select__k` v grid searchi: 5, 8, 10, 12,
    - cieľ: nájsť vhodný kompromis medzi dimenziou a výkonom.

---

## 5. Hodnotenie modelov a validácia

### 5.1 Train/test split

- `train_test_split(X, y, test_size=..., random_state=...)`:
  - oddeľuje trénovaciu a testovaciu množinu,
  - testovacia množina sa nepoužíva pri tréningu ani pri výbere hyperparametrov,
  - slúži len na finálne zhodnotenie generalizačného výkonu.

### 5.2 Cross-validation (CV)

- **KFold(n_splits=k)**:
  - rozdelí dataset na k častí (foldov),
  - iteratívne:
    - 1 fold = validácia,
    - k-1 foldov = tréning,
  - každá vzorka je raz v testovacom folde.
- V cvičeniach:
  - ukázané cez `KFold(n_splits=5)` a výpis počtu vzoriek vo foldoch.

- **cross_val_score(estimator, X, y, cv=k)**:
  - automaticky vykoná k-fold CV,
  - vráti vektor skóre (accuracy/R²) pre každý fold,
  - priemer týchto skóre dáva stabilnejší odhad generalizačnej chyby než iba jeden train/test split.

### 5.3 Hyperparameter tuning – Grid search

#### Ručný grid

- pomocou `itertools.product` nad zoznamami možných hodnôt hyperparametrov
- v cvičeniach:
  - DecisionTree: kombinácie `criterion` × `max_features`,
  - ručný výpočet accuracy a vizualizácia heatmapy.

#### `GridSearchCV`

- automatizovaný grid search s cross-validation:
  - `GridSearchCV(estimator, param_grid, cv, scoring, n_jobs, ...)`.
- kľúčové atribúty:
  - `best_params_` – najlepšia kombinácia hyperparametrov,
  - `best_score_` – najlepšie priemerné CV skóre,
  - `best_estimator_` – model/pipeline s najlepšími nastaveniami,
  - `cv_results_` – detailné výsledky pre všetky kombinácie.

- V cvičeniach:
  - SVC + StandardScaler (klasifikácia na vlastných dátach – výber C, gamma, kernel, degree),
  - SVR + SelectKBest + StandardScaler (regresia na Boston housingu – výber K, kernel, C, gamma),
  - Iris: `SimpleImputer` + scaler + (DecisionTree alebo LogisticRegression) – grid pre viac modelov v jednom behu.

### 5.4 Pipeline

- `Pipeline([('krok1', transformátor1), ('krok2', transformátor2), ('clf', klasifikátor)])`.
- Význam:
  - zaručuje správne poradie operácií (napr. scale až po imputácii),
  - integrovateľný s `GridSearchCV`,
  - v CV sa pre každý fold fitujú znova aj preprocessing kroky (žiadny leak informácií z testu do tréningu).
- Referencovanie parametrov v `param_grid`:
  - `krok__parameter`, napr.:
    - `clf__C`, `clf__gamma`, `select__k`, `imputer__strategy`, `scaler`.

- V Iris cvičení: multi-model pipeline:
  - v `param_grid` sú dve vetvy (zoznam dvoch dictov):
    - jedna pre DecisionTree,
    - druhá pre LogisticRegression → GridSearchCV robí aj model selection.

---

## 6. Vizualizácie

- **Mriežka obrázkov tvárí**:
  - vizualizácia datasetu Olivetti faces,
  - funkcia `print_faces(images, target, top_n)` – zobrazenie niekoľkých tvárí + ich ID.

- **Decision boundary pre SVM**:
  - generovanie gridu bodov v 2D,
  - predikcia tried pre každý bod,
  - vykreslenie bodov farebne podľa predikcie + skutočné vzorky.

- **Accuracy vs. max_depth** (Decision Tree):
  - graf závislosti presnosti od maximálnej hĺbky stromu,
  - ilustruje underfitting (príliš malá hĺbka) vs overfitting (príliš veľká hĺbka).

- **Heatmapa accuracy**:
  - rozhodovací strom: kombinácie `criterion` × `max_features`,
  - farba reprezentuje priemernú presnosť.

- **Scatter: feature vs target** (Boston):
  - pre prvých pár featur graf (X feature, Y target),
  - vizuálna kontrola, ktoré featury korelujú s cenou domov.

- **Predicted vs actual (regresia)**:
  - Boston housing:
    - x: skutočné ceny,
    - y: predikované ceny,
    - diagonála = ideálna predikcia,
    - rozptyl okolo diagonály = chyba modelu.

---

## 7. Stručné zhrnutie – čo ovládať z prvej časti

1. Rozdiel medzi klasifikáciou a regresiou, typické metriky (accuracy, R²).
2. Základné modely: SVC/SVR, DecisionTree, RandomForest, LogisticRegression – účel + kľúčové hyperparametre.
3. Predspracovanie: škálovanie (StandardScaler, MinMaxScaler), imputácia (SimpleImputer), kódovanie kategórií (LabelEncoder, OneHotEncoder), feature selection (SelectKBest).
4. Delenie dát: train/test split vs cross-validation (KFold, cross_val_score).
5. Hyperparameter tuning:
   - ručný grid vs `GridSearchCV`,
   - interpretácia `best_params_`, `best_score_`, `best_estimator_`.
6. Pipeline:
   - dôvod použitia,
   - referencovanie parametrov krokov (`krok__parameter`),
   - multi-model grid search (rôzne modely v jednom `GridSearchCV`).
7. Základná interpretácia vizualizácií (decision boundary, heatmapy, scattery).

