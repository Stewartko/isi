# ISI – Zápočet (1. + 2. časť) – Tréningové otázky

Scenár: 10 otázok / 30 minút.  
Štruktúra podľa zadania:
- **5 otázok – stručné doplnenie kódu** (názov funkcie/triedy, 1 riadok kódu),
- **5 otázok – identifikácia algoritmu + krátka teória + prípadné doplnenie riadku**  
  (BFS, DFS, Greedy, A*, MRV, LCV, plus základy z 1. časti).

Penalizácia: za nesprávnu odpoveď 0 bodov (žiadne mínusové body).  
Žiadne CSP modelovanie ani constraint sekcie (MRV/LCV len ako heuristiky).

---

## ČASŤ A – Doplnenie kódu (5 otázok)

### Otázka A1 – Rozdelenie dát na trénovaciu a testovaciu množinu

Doplň riadok s volaním `train_test_split`, aby sa dáta rozdelili v pomere 75 % train, 25 % test, s `random_state=0`.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=1000, 
                                    n_features=3, n_redundant=0)

# DOPLŇ TENTO RIADOK:
X_train, X_test, y_train, y_test = ...
```

**Riešenie A1:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
```

---

### Otázka A2 – Výber klasifikátora a vyhodnotenie presnosti

Doplň:
1. vytvorenie **rozhodovacieho stromu**,
2. výpočet **presnosti klasifikácie** pomocou `accuracy_score`.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# predpokladaj, že X_train, X_test, y_train, y_test už existujú

# 1) DOPLŇ: vytvor klasifikátor dtc
dtc = ...

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

# 2) DOPLŇ: vypočítaj presnosť klasifikácie
acc = ...
print("Presnosť:", acc)
```

**Riešenie A2:**

```python
dtc = DecisionTreeClassifier()

acc = accuracy_score(y_test, y_pred)
```

---

### Otázka A3 – Normalizácia/škálovanie prvých troch príznakov

Doplň použitie `StandardScaler`, aby sa prvé tri stĺpce matice `X` (napr. z Boston datasetu) skonvertovali na škálované `X_scaled` so stredom 0 a smerodajnou odchýlkou 1.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# predpokladaj, že X je ndarray shape (n_samples, n_features)

scaler = StandardScaler()

# DOPLŇ: fit + transform na prvé tri stĺpce
X_scaled_first3 = ...
```

**Riešenie A3:**

```python
X_scaled_first3 = scaler.fit_transform(X[:, :3])
```

---

### Otázka A4 – Imputácia chýbajúcich hodnôt v Iris pomocou mediánu

Doplň vytvorenie `SimpleImputer` so stratégiou *median* a jeho použitie na `iris_X`.

```python
import numpy as np
from sklearn import datasets
from sklearn.impute import SimpleImputer

iris = datasets.load_iris()
iris_X = iris.data

# predpokladaj, že sme už doplnili náhodné NaN hodnoty

# DOPLŇ: imputer s mediánom
imputer = ...

# DOPLŇ: transformácia
iris_X_imp = ...
```

**Riešenie A4:**

```python
imputer = SimpleImputer(strategy="median")
iris_X_imp = imputer.fit_transform(iris_X)
```

---

### Otázka A5 – GridSearchCV s pipeline (škálovanie + SVC)

Doplň `param_grid` pre SVC v pipeline:  
- meníme parameter `C` v logaritmickom rozsahu (napr. `[0.1, 1, 10]`),  
- skúšame len `rbf` kernel.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe_svc = Pipeline([("scl", StandardScaler()),
                     ("clf", SVC())])

# DOPLŇ param_grid tak, aby menil C a kernel
param_grid = [
    {
        ...
    }
]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring="accuracy",
                  cv=5)
```

**Riešenie A5:**

```python
param_grid = [
    {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf"]
    }
]
```

---

## ČASŤ B – Algoritmy a heuristiky (5 otázok)

### Otázka B1 – Rozpoznanie BFS v pseudokóde

Máš nasledujúci pseudokód (pre 8-puzzle):

```text
fronta := [start]
visited := ∅

while fronta nie je prázdna:
    s := fronta.pop_front()
    ak s je cieľ:
        vráť riešenie
    pridaj s do visited
    pre každý nasledovník n stavu s:
        ak n nie je vo visited ani vo fronte:
            fronta.push_back(n)
```

1. O aký algoritmus ide?  
2. Akú dátovú štruktúru používa na otvorené stavy?  
3. Je tento algoritmus optimálny pri jednotkovej cene hrán?  
4. Je možné ho zapísať rekurzívne?

**Riešenie B1:**  
1. **BFS – Breadth-First Search.**  
2. Používa **frontu (FIFO)** pre otvorené stavy.  
3. Pri jednotkovej cene hrán je **optimálny** (nájde najkratšiu cestu v počte krokov).  
4. Teoreticky áno, ale štandardná implementácia je **iteratívna**; rekurzívna verzia BFS nie je prirodzená a vo výuke sa bežne nepoužíva.

---

### Otázka B2 – Rozpoznanie DFS v pseudokóde

Pseudokód:

```text
zásobník := [start]
visited := ∅

while zásobník nie je prázdny:
    s := zásobník.pop()
    ak s je cieľ:
        vráť riešenie
    ak s nie je vo visited:
        pridaj s do visited
        pre každý nasledovník n stavu s:
            zásobník.push(n)
```

1. O aký algoritmus ide?  
2. Akú dátovú štruktúru používa na otvorené stavy?  
3. Je optimálny pri jednotkovej cene hrán?  
4. Je prirodzene implementovateľný rekurzívne?

**Riešenie B2:**  
1. **DFS – Depth-First Search.**  
2. Používa **zásobník (LIFO)**.  
3. Nie, **nie je optimálny** – môže nájsť dlhšiu cestu, aj keď existuje kratšia.  
4. Áno, DFS je prirodzene implementovateľný rekurzívne (rekurzia zodpovedá zásobníku).

---

### Otázka B3 – Greedy best-first search vs. A*

Daj stručné odpovede:

1. Ako sa rozhoduje **Greedy best-first search**, ktorý stav expandovať?  
2. Ako sa rozhoduje **A\***, ktorý stav expandovať?  
3. Ktorý z nich je pri admisibilnej heuristike garantovane optimálny?  
4. Akú úlohu má funkcia **h(n)** a **g(n)**?

**Riešenie B3:**  
1. Greedy best-first search expanduje stav s **najnižšou heuristickou hodnotou h(n)** – ignoruje cenu g(n).  
2. A* expanduje stav s **najnižšou hodnotou f(n) = g(n) + h(n)**.  
3. Pri admisibilnej (a konzistentnej) heuristike je **A\*** optimálny. Greedy best-first nie je garantovane optimálny.  
4. **h(n)** odhaduje vzdialenosť do cieľa, **g(n)** je doterajšia cena cesty od štartu k n (napr. počet krokov).

---

### Otázka B4 – MRV a LCV (heuristiky)

Stručne vysvetli:

1. Čo robí heuristika **MRV (Minimum Remaining Values)** pri výbere premennej v hľadaní riešenia?  
2. Čo robí heuristika **LCV (Least Constraining Value)** pri výbere hodnoty pre danú premennú?  
3. Ako MRV a LCV typicky ovplyvňujú počet expandovaných stavov?

**Riešenie B4:**  
1. **MRV** vyberá premennú, ktorá má **najmenšiu doménu** (najmenej legálnych hodnôt) – t. j. je „najviac obmedzená“.  
2. **LCV** zoradí hodnoty tak, aby sa najprv skúšali tie, ktoré **najmenej obmedzia** možnosti ostatných premenných (susedov).  
3. MRV aj LCV zvyčajne **znižujú počet expandovaných stavov** – MRV zlyhá skôr pri nerešiteľných vetvách, LCV zasa udržuje viac možností pre budúce rozhodnutia.

---

### Otázka B5 – Identifikácia algoritmu z kódu (A* na 8-puzzle)

Maj nasledujúci (skrátený) Python kód:

```python
def search(start_state, heuristic):
    import heapq, itertools

    open_set = []
    counter = itertools.count()

    g_score = {start_state: 0}
    f_score = {start_state: heuristic(start_state)}

    heapq.heappush(open_set, (f_score[start_state], next(counter), start_state))

    while open_set:
        _, _, state = heapq.heappop(open_set)

        if state.is_goal():
            return state

        for move in state.get_possible_moves():
            new_state = state.move_tile(move)
            tentative_g = g_score[state] + 1

            if tentative_g < g_score.get(new_state, float("inf")):
                g_score[new_state] = tentative_g
                f_score[new_state] = tentative_g + heuristic(new_state)
                heapq.heappush(open_set, (f_score[new_state], next(counter), new_state))
```

1. O aký algoritmus ide?  
2. Akú dátovú štruktúru používa `open_set`?  
3. Ako je definovaná výberová funkcia f(n)?  
4. Ktorú jednu vec by si musel doplniť, aby kód neotváral zbytočne tie isté stavy opakovane? (stručne slovne).

**Riešenie B5:**  
1. Ide o **A\*** (A-star) vyhľadávanie.  
2. `open_set` je **prioritný rad (min-heap)** implementovaný cez `heapq`.  
3. f(n) = g(n) + h(n), kde:
   - `g_score[new_state]` je cena cesty od štartu,
   - `heuristic(new_state)` je odhad vzdialenosti do cieľa.  
4. Potrebné je zaviesť evidenciu **uzavretých/visited stavov** (napr. množinu `closed_set`), aby sa už raz spracované stavy neexpanovali znova – pri popnutí zo zásobníka sa stav pridá do `closed_set` a noví nasledovníci sa nepridávajú, ak už v `closed_set` sú.

---

Tieto otázky pokrývajú typ úloh, ktorý je avizovaný pre zápočet:
- doplnenie jednoduchých, ale významných riadkov kódu (1. časť – sklearn, dátové prípravy, grid search),
- rozpoznanie a charakteristika algoritmov BFS/DFS/Greedy/A*, plus heuristík MRV/LCV (2. časť – vyhľadávanie, heuristiky).
