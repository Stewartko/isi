# Poznámky k skúške: Algoritmy a Machine Learning (výberové otázky)

> Zamerané na: vyhľadávanie v grafoch (BFS/DFS/UCS/A*/greedy), CSP (backtracking, forward checking, arc-consistency, min-conflicts), lokálne a evolučné heuristiky (hill climbing, simulated annealing, genetické algoritmy), základné ML modely (SVM, perceptrón, rozhodovacie stromy, random forest), validácia a predspracovanie dát.

---

## 1) Validácia a generalizácia

### Cross-validation (krížová validácia)
- Účel: **odhad výkonu na nevidených dátach** (generalizácia) + výber/tuning hyperparametrov.
- **K-fold CV**: dáta sa rozdelia na K častí (foldov); K-krát trénujem na K−1 foldoch a testujem na 1 folde; výsledok spriemerujem.
- **Leave-One-Out (LOOCV)**: špeciálny prípad, keď **K = počet vzoriek** (každý fold má 1 vzorku).
- **Hold-out**: iba jedno rozdelenie (train/test).

### Stratified (stratifikovaná) cross-validation
- Použitie: hlavne pri **nevyvážených triedach**.
- Vlastnosť: každý fold má **podobný pomer tried** ako celý dataset.

### Learning curve (krivka učenia)
- Zobrazuje tréningové a validačné skóre/chybu **v závislosti od veľkosti tréningovej množiny**.
- Diagnostika:
  - **overfitting**: tréningová chyba nízka, validačná výrazne vyššia.
  - **underfitting**: obe chyby relatívne vysoké a blízko seba.

### Validation curve (krivka validácie)
- Zobrazuje tréningové a validačné skóre/chybu **v závislosti od hodnoty hyperparametra** (napr. C v SVM, K v KNN, max_depth v strome).

---

## 2) Overfitting vs Underfitting

### Overfitting (pretrénovanie)
- Model sa „naučí“ aj šum; **výborný na tréningu**, horší na validácii/teste.
- Typický signál: **veľký rozdiel** train vs val chyba.

### Underfitting (podtrénovanie)
- Model je príliš jednoduchý alebo príliš regularizovaný; nezachytí vzory.
- Typický signál: train aj val chyba **vysoké** a podobné.

---

## 3) Zhlukovanie (Clustering)

### K-means
- **K = počet zhlukov (clusterov)** (to je význam hyperparametra).
- Idea: opakovane priraďuj body k najbližšiemu centroidu a centroidy prepočítaj ako priemer.
- Centroid = „stred“ zhluku = **aritmetický priemer** bodov v klastri (po dimenziách).
- Minimalizuje sa (intuícia): **súčet štvorcov vzdialeností** bodov od centroidu (within-cluster SSE).
- Výber K:
  - **Elbow (lakťová) metóda**: hľadáš bod, kde zlepšenie SSE už výrazne spomaľuje.
  - **Silhouette (siluetová) metóda**: hodnotí, ako dobre sú body „v svojom“ klastri vs mimo.

### DBSCAN
- Parametre:
  - **ε (epsilon)**: polomer okolia (radius).
  - **MinPts**: minimálny počet bodov v ε-okolí, aby bol bod jadrový (core).
- Pojmy:
  - **core point**: má ≥ MinPts bodov v ε-okolí.
  - **border point**: je v ε-okolí core bodu, ale sám nemá MinPts.
  - **noise/outlier**: nepatrí do žiadneho klastru.
- „Density“ (hustota): prakticky **počet bodov v danom ε-okolí**.

---

## 4) Vyhľadávanie v grafoch (Search)

### BFS (Breadth-First Search) – do šírky
- Dátová štruktúra: **FIFO fronta**.
- Expanduje po úrovniach (hĺbka 0, potom 1, potom 2, …).
- Vlastnosti:
  - **kompletný** (ak je priestor konečný a vetvenie rozumné),
  - **optimálny** pri rovnakých váhach hrán (alebo keď minimalizuješ počet krokov).
- Nie je optimálny pri rôznych váhach hrán.

### DFS (Depth-First Search) – do hĺbky
- Dátová štruktúra: **zásobník / rekurzia**.
- Ide čo najhlbšie, potom sa vracia (backtracking).
- Vlastnosti:
  - nižšia pamäť než BFS (typicky drží hlavne cestu),
  - **nezaručuje najkratšiu cestu**,
  - v grafe s cyklami sa môže zacykliť bez „visited“,
  - nie je kompletný v nekonečne hlbokom priestore (bez limitu hĺbky).

### UCS (Uniform Cost Search) – uniformné náklady
- Dátová štruktúra: **prioritná fronta podľa g(n)**.
- Vždy expanduje uzol s najmenšou **kumulatívnou cenou od štartu**:
  - \( g(n) = \) najlacnejšia nájdená cena zo štartu do n.
- Vlastnosti:
  - **optimálny** pri nezáporných váhach (je to Dijkstra pre „do cieľa“).
  - Nerobí heuristiku (na rozdiel od A*).

### Greedy best-first search (v kontexte hľadania cesty)
- Typicky vyberá uzol podľa **heuristiky h(n)** (ignoruje g(n)).
- „Zanedbáva reálne váhy od štartu“ → často rýchly, ale môže byť neoptimálny a aj nekompletný.

### A* (A-star)
- Kriteriálna funkcia:
  - \( f(n) = g(n) + h(n) \)
- Vlastnosti:
  - používa heuristiku **h(n)** (odhad z n do cieľa),
  - expanduje uzly s najnižším **f(n)**,
  - **optimálny**, ak je heuristika **prípustná (admissible)** (a pri grafe sa typicky vyžaduje konzistentnosť pre jednoduchšiu správnosť s uzatváraním uzlov).

### Prípustná (admissible) heuristika
- Definícia: heuristika nikdy **nepreceňuje** skutočný minimálny zvyšný náklad:
  - \( h(n) \le h^*(n) \)
- Dôsledok: A* je optimálny (pri vyhľadávaní v strome; v grafe za bežných podmienok tiež, ak sa korektne spracuje „closed set“).

### Konzistentná (consistent / monotónna) heuristika
- Definícia (trojuholníková nerovnosť):
  - \( h(n) \le c(n,n') + h(n') \) pre každú hranu \( n\to n' \)
- Dôsledok: hodnota \( f(n)=g(n)+h(n) \) **po hrane neklesá** (je neklesajúca).
- Konzistentnosť **implikuje prípustnosť**.

### Obojsmerné vyhľadávanie (Bidirectional search)
- Súčasne hľadá:
  - dopredu zo štartu a dozadu z cieľa, kým sa fronty nestretnú.
- Efektívne najmä keď:
  - poznáš **štart aj cieľ**,
  - priestor je veľký a vetvenie vysoké,
  - vieš generovať prechody aj „odzadu“ (reverzibilné/definované).

### Iterative Deepening (IDDFS)
- Kombinuje BFS (optimálnosť v krokoch) a DFS (pamäť):
  - spúšťa DFS s limitom hĺbky 0,1,2,... až nájde cieľ.
- Výhoda: pamäť ako DFS, ale nájde riešenie v najmenšej hĺbke ako BFS (pri jednotkových krokoch).

---

## 5) CSP (Constraint Satisfaction Problems)

### Backtracking
- Systematické hľadanie priradenia:
  - vyber premennú → skús hodnotu → ak konflikt, vráť sa (dead end) a skús inú.
- Základná úloha: nájsť **konzistentné priradenie** všetkých premenných.

### Forward checking (dopredná kontrola)
- Po priradení hodnoty pre premennú:
  - „pozrie dopredu“ na susedné nepřiradené premenné a **preriedi ich domény**.
- Cieľ: znížiť počet spätných krokov (backtrackingov).

### Arc-consistency (hranová konzistencia)
- Pre každú dvojicu premenných (X, Y) s obmedzením:
  - každá hodnota v doméne X musí mať **aspoň jednu podporu** v doméne Y, aby bola kompatibilná s obmedzením.
- Prakticky: **odstraňuje nekonzistentné hodnoty z domén**, čím zmenšuje vyhľadávací priestor.
- Negarantuje optimum (CSP je satisfakčný problém), ale výrazne pomáha efektivite.

### Min-conflicts
- Lokálne vyhľadávanie pre CSP:
  1) začne **úplným** (často náhodným) priradením,
  2) kým existujú konflikty: vyber premennú v konflikte,
  3) nastav jej hodnotu tak, aby **minimalizovala počet konfliktov** (pri zhode náhodne).
- Veľmi dobré napr. pre N-queens a veľké „rozvrhové“ CSP.

---

## 6) Lokálne vyhľadávanie a metaheuristiky

### Hill Climbing (horolezecká metóda)
- Vždy prejde do lepšieho suseda podľa evaluačnej funkcie.
- Nevyžaduje znalosť celého priestoru (len susedov).
- Problémy:
  - **lokálne optimum**: všetci susedia horší, ale globálne riešenie je inde.
  - **plateau (planina)**: susedia majú **rovnakú** (alebo takmer rovnakú) hodnotu.
  - „ridge/saddle“: komplikovaný tvar terénu.
- Typické opravy: random restart, sideways moves (obmedzený pohyb po planine), simulované žíhanie.

### Simulated Annealing (simulované žíhanie)
- Podobné hill climbing, ale občas prijme aj horší krok, aby unikol z lokálneho optima.
- Teplota \(T\) riadi ochotu robiť „zlé“ ťahy:
  - vysoké T: viac náhody,
  - nízke T: správa sa skoro ako greedy/hill climbing.
- Pri minimalizácii s \(\Delta = \text{new}-\text{old}\):
  - ak \(\Delta \le 0\): prijmi,
  - ak \(\Delta > 0\): prijmi s pravdepodobnosťou \(p = e^{-\Delta/T}\).

### Genetické algoritmy (GA)
- Populácia kandidátov (chromozóm = kód riešenia).
- Kroky:
  1) inicializuj populáciu,
  2) ohodnoť (fitness),
  3) selekcia rodičov,
  4) kríženie (crossover),
  5) **mutácia**,
  6) nová generácia, opakuj.
- Mutácia:
  - kľúčová úloha: **udržať genetickú diverzitu** (náhodná zmena génu), aby populácia „nezamrzla“ v lokálnom optime.
- GA sú heuristiky: **negenerujú garanciu globálneho optima**, ale často nájdu dobré riešenia v obrovských priestoroch.

---

## 7) ML modely a pojmy

### Perceptrón
- Jednoduchý lineárny klasifikátor:
  - \( y = \text{step}(w^T x + b) \)
- Ak dáta nie sú lineárne separovateľné:
  - klasický perceptrón **nemusí konvergovať**.
- Typická aktivačná funkcia: **skoková (step)**.

### SVM (Support Vector Machine)
- Hľadá deliacu hyperrovinu s **maximálnym marginom**.
- Soft margin:
  - umožní niektoré chyby pomocou slack premenných \(\xi\),
  - parameter **C** nastavuje kompromis medzi šírkou marginu a penalizáciou chýb:
    - veľké C → menej chýb na tréningu, vyššie riziko overfittingu,
    - malé C → viac tolerancie, hladšia hranica (môže underfit).
- Kernelizácia (kernel trick):
  - rieši **lineárne neseparovateľné** dáta tým, že efektívne pracuje v „vyššej dimenzii“.

### Rozhodovacie stromy a Information Gain
- Atribút (feature): premenná, podľa ktorej delíš.
- Uzol: miesto v strome (root, interný uzol, list).
- Information Gain: používa sa na výber atribútu/splittu v uzle:
  - \( IG = H(\text{parent}) - \sum_i p_i H(\text{child}_i) \)
  - kde \(H\) je entropia (alebo iná nečistota).

### Random Forest (náhodný les)
- Ensemble: kombinuje predikcie **mnohých stromov**.
- Bagging:
  - každý strom trénuje na bootstrap vzorke dát,
  - pri splitovaní sa často berie len náhodná podmnožina feature.
- Efekt: znižuje varianciu (stabilnejšie než jeden strom).

---

## 8) Predspracovanie dát

### Škálovanie (scaling) – prečo
- Cieľ: aby mali feature podobný rozsah, inak algoritmy citlivé na mierku (KNN, K-means, SVM, gradientné metódy) „preferujú“ veľké čísla.

### Normalizácia vs štandardizácia
- Normalizácia (Min-Max):
  - \( x' = \frac{x-\min(x)}{\max(x)-\min(x)} \) → typicky do [0,1].
- Štandardizácia (Z-score):
  - \( z = \frac{x-\mu}{\sigma} \) → priemer 0, odchýlka 1.

### Kategorické dáta: Label encoding vs One-Hot Encoding (OHE)
- Label encoding: mapuje kategórie na čísla (A=0,B=1,C=2) → môže zaviesť **falošné poradie/vzdialenosť**.
- OHE: každá kategória je vlastný binárny stĺpec → bez umelého poradia (za cenu vyššej dimenzie).

### Imputácia
- Účel: spracovať **chýbajúce hodnoty** (mean/median/mode/constant).
- scikit-learn: **SimpleImputer**.

---

## 9) Feature selection a ladenie hyperparametrov

### Feature selection – prečo
- Znížiť overfitting, zrýchliť tréning, zlepšiť interpretáciu.
- Typy:
  - Filter (štatistické kritériá),
  - Wrapper (testuje subsety pomocou ML modelu),
  - Embedded (výber počas učenia, napr. L1).

### Wrapper metóda
- „Balenie“: hodnotí subsety feature tak, že **trénuje model** a meria výkon (napr. RFE, forward/backward selection).

### Grid search
- Metóda ladenia hyperparametrov:
  - prejde mriežku kombinácií a vyberie najlepšiu (typicky s CV).

---

## 10) Metriky (rýchle)

### Precision
- \(\text{Precision} = \frac{TP}{TP+FP}\)
- „Zo všetkých pozitívnych predikcií, koľko bolo správne pozitívnych?“

---

## 11) Agenti v AI (rýchle)

### Reflexný agent
- Rozhoduje sa len podľa **aktuálneho vnímania/stavu** (typicky pravidlá IF–THEN).
- Neplánuje dopredu a typicky neudržiava bohatý interný model sveta.

### Racionálny agent
- Vyberá akcie tak, aby **maximalizoval očakávaný úžitok** (given percepts + znalosti).

---

## 12) Najčastejšie skúškové „pasce“ (mini-checklist)
- K-means: **K je počet klastrov** (nie metóda na určenie K).
- BFS: optimálny len pri rovnakých váhach / krokoch.
- UCS: optimálny pri nezáporných váhach; prioritná fronta podľa g(n).
- A*: f=g+h; admissible = nepreceňuje; consistent = trojuholníková nerovnosť.
- Greedy best-first: často používa len h(n) (ignoruje g(n)).
- Hill climbing: nepreskakuje „dole“ → uviazne; plateau = rovnaké hodnoty.
- Simulated annealing: vie prijať horší krok (pravdepodobnostne).
- GA: mutácia = diverzita; crossover = kombinovanie rodičov.
- SVM: kernel = nelineárne hranice; C veľké → riziko overfittingu.
- Label encoding môže vniesť poradie; OHE to rieši (za cenu dimenzie).
- Imputácia = dopĺňanie chýbajúcich hodnôt.

---
