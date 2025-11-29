# Druhá časť – Teória (od cvičenia 7)

## Cvičenie 7 – Vyhľadávanie v stavovom priestore (8‑puzzle)

### 1. Formulácia problému 8‑puzzle

- **Stav**: konfigurácia dosky N‑puzzle (typicky 3×3, teda 8‑puzzle).  
  - Doska obsahuje čísla 1..8 a `0` (prázdne políčko).  
  - Stav je úplná informácia o rozložení dielikov, napr. 2D matica alebo zoznam dĺžky 9.
- **Počiatočný stav**: zadané rozloženie dielikov (často permutácia cieľa).
- **Cieľový stav**: dieliky 1..8 v poradí zľava doprava, hore dole, `0` na poslednej pozícii:  
  `1 2 3 / 4 5 6 / 7 8 0`.
- **Operátory**: posunutie dielika do prázdneho políčka (0) – t. j. zámenná `0` so susedom vľavo/vpravo/hore/dole (ak existuje).  
- **Stavový priestor**: všetky dosiahnuteľné konfigurácie dosky pre danú počiatočnú konfiguráciu.

### 2. Trieda reprezentujúca stav (PuzzleState – koncept)

- Uchováva:
  - `board` – 2D štruktúru (matica) s hodnotami 0..N‑1,
  - `size` – veľkosť dosky (3 pre 3×3),
  - `parent` – odkaz na predchádzajúci stav (na rekonštrukciu riešenia),
  - `moved_tile` – číslo dielika, ktorý bol posunutý z rodiča do aktuálneho stavu,
  - pozíciu prázdneho políčka (napr. cache `_blank_pos`).

- Kľúčové operácie:
  - **generovanie nasledovníkov**: `get_possible_moves()` → zoznam dielikov, ktoré je možné posunúť do prázdneho políčka (susedia 0),
  - **aplikácia ťahu**: `move_tile(tile)` → nový stav so zameneným `0` a `tile`,
  - **test cieľa**: `is_goal()` → porovnanie aktuálnej konfigurácie s cieľovou,
  - **rekonštrukcia cesty**: `reconstruct_path()` → ide po `parent` smerom k štartu a zbiera `moved_tile`,
  - **hash/rovnosť**: implementácia `__eq__` + `__hash__` → stav je použiteľný v `set` a ako kľúč v slovníkoch (navštívené stavy).

### 3. Heuristiky pre 8‑puzzle

- **Heuristika h₁ – počet zle umiestnených dielikov (misplaced tiles)**:
  - Počíta, koľko dielikov nie je na svojej cieľovej pozícii (0 sa ignoruje).
  - Jednoduchá, ale pomerne slabá.
- **Heuristika h₂ – Manhattanovská vzdialenosť (Manhattan distance)**:
  - Pre každý dielik zistí jeho cieľovú pozíciu a spočíta \\(|r - r_{cieľ}| + |c - c_{cieľ}|\\).
  - Výsledkom je súčet Manhattan vzdialeností všetkých dielikov (okrem 0).
- **Admisibilita**:
  - Heuristika je **admisibilná**, ak **nikdy nepreceňuje** skutočnú vzdialenosť do cieľa.
  - Pre 8‑puzzle je Manhattanovská vzdialenosť štandardne považovaná za admisibilnú.
- **Konzistentnosť (monotónnosť)**:
  - Hovorí, že h(n) ≤ cost(n, n') + h(n') pre každú hranu (n → n'),
  - pri konzistentnej heuristike má A* pekné vlastnosti (netreba znovu otvárať uzly).

### 4. Algoritmy vyhľadávania

#### 4.1 BFS – Breadth‑First Search

- Používa **frontu (FIFO)**.  
- Princíp:
  - začína v počiatočnom stave, ten vloží do fronty,
  - opakovane vyberá prvý stav z fronty:
    - ak je cieľový → koniec,
    - inak generuje všetkých nasledovníkov a všetkých, ktorí ešte neboli v `visited` ani vo fronte, pridá na koniec fronty.
- **Vlastnosti**:
  - pri jednotkovej cene hrán (každý krok má cenu 1) je BFS **optimálny** (nájde cestu s najmenším počtom krokov),
  - je **kompletný** (ak existuje riešenie, nájde ho),
  - má veľké pamäťové nároky (v pamäti drží celú „vlnu“ fronty).

#### 4.2 DFS – Depth‑First Search

- Používa **zásobník (LIFO)**.
- Princíp:
  - vždy expanduje naposledy vložený stav („chodí do hĺbky“),
  - pri obišielom strome/grafe môže zbehnúť veľmi hlboko v jednej vetve.
- **Vlastnosti**:
  - pamäťovo úspornejší ako BFS (drží len cestu + niekoľko vetiev),
  - **nie je optimálny** (môže nájsť dlhšiu cestu, aj keď kratšia existuje),
  - **nemusí byť kompletný** v nekonečnom grafe alebo bez obmedzenia hĺbky (môže sa „stratiť“ v nekonečnej vetve),
  - často sa používa limit (max. počet navštívených stavov alebo max. hĺbka).

#### 4.3 Greedy best‑first search

- Používa **prioritný rad (min‑heap)** podľa heuristiky h(n).  
- Princíp:
  - fronta je usporiadaná podľa h(n) – vždy expanduje stav, ktorý **vyzerá najbližšie k cieľu** podľa heuristiky,
  - **neberie do úvahy** skutočnú cenu g(n) (počet krokov od štartu).
- **Vlastnosti**:
  - môže byť rýchly (najmä pri dobrej heuristike),
  - **nie je optimálny** – heuristika môže zobrať „lokálne sľubnú“, ale globálne zlú vetvu.

#### 4.4 A* (A‑star)

- Používa prioritný rad podľa **f(n) = g(n) + h(n)**:  
  - g(n) = doterajšia cena od počiatočného stavu (počet krokov pri jednotkových nákladoch),
  - h(n) = heuristický odhad vzdialenosti do cieľa,
  - f(n) = odhad celkovej ceny cesty cez n.
- Uchováva mapy:
  - `g_score[state]` – najlepšia doteraz nájdená cena pre stav,
  - `f_score[state] = g_score[state] + h(state)`.
- **Vlastnosti**:
  - pri **admisibilnej** a konzistentnej heuristike je A* **optimálny** a **kompletný**,
  - v porovnaní s BFS často výrazne redukuje počet expandovaných stavov, najmä na veľkých stavových priestoroch.

### 5. Metriky: discovered, expanded

- **discovered** – koľko stavov bolo vložených do fronty/stacku/haldy (koľko stavov sme „objavili“),
- **expanded** – koľko stavov sme reálne **expandovali** (vybrali z fronty/stacku/haldy a vygenerovali z nich nasledovníkov),
- slúžia na porovnanie efektivity rôznych algoritmov a heuristík.

---

## Cvičenie 8 – Sudoku ako CSP (Constraint Satisfaction Problem)

### 1. Formulácia Sudoku ako CSP

- **Premenné**: 81 buniek Sudoku (r, c) pre r,c ∈ {0..8}.
- **Domény**:
  - pre každú bunku množina možných hodnôt {1,..,9},
  - pri čiastočne vyplnenej úlohe sú niektoré bunky vopred priradené, iné majú domény zúžené obmedzeniami.
- **Obmedzenia**:
  - každý **riadok** musí obsahovať čísla 1..9 bez opakovania,
  - každý **stĺpec** musí obsahovať čísla 1..9 bez opakovania,
  - každý **3×3 blok** musí obsahovať čísla 1..9 bez opakovania.

### 2. Brute‑force DFS vs. CSP backtracking

- **Brute‑force DFS**:
  - vypĺňa mriežku v pevnom poradí,
  - skúša všetky hodnoty 1..9 bez priebežného testovania konzistencie,
  - až po úplnom vyplnení kontroluje korektnosť celej mriežky,
  - extrémne neefektívny (obrovský počet neplatných medzistavov).
- **CSP backtracking**:
  - pri každom priradení používame `is_valid(r, c, val)` (okamžitá kontrola obmedzení),
  - kombinujeme s heuristikami (MRV, LCV) a technikami ako Forward Checking,
  - výrazne redukuje vyhľadávací priestor.

### 3. Heuristiky pre CSP

#### 3.1 MRV – Minimum Remaining Values

- Vyberá premennú (bunku) s **najmenším počtom legálnych hodnôt** v doméne.
- Intuícia:
  - ak má premenná málo možností, je najkritickejšia,
  - ak je problém neriešiteľný, často sa to prejaví na týchto „najviac obmedzených“ premenných,
  - skôr odhalíme slepé vetvy.

#### 3.2 LCV – Least Constraining Value

- Pre vybranú bunku:
  - hodnoty v doméne zoradíme podľa toho, ako **málo obmedzujú** ostatné premenné (susedné bunky v riadku, stĺpci a bloku),
  - preferujeme hodnoty, ktoré „neberú“ veľa možností ostatným – nechávame viac flexibility do budúcna.

### 4. Forward Checking

- Po priradení hodnoty do bunky (napr. `(r, c) = val`) spraví **look‑ahead**:
  - pre každú ešte nepriradenú bunku spočíta aktuálnu doménu (možné hodnoty),
  - ak **niektorá doména je prázdna**, znamená to, že v budúcnosti nebude možné všetko korektne vyplniť – vetvu môžeme okamžite „odrezať“ (backtracking).
- Výhoda:
  - slepé vetvy sa odhalia skôr, než sa príliš prehĺbime,
  - znižuje počet expandovaných stavov.

### 5. Spätné prehľadávanie (backtracking) so
- / bez heuristík

- **Základný backtracking**:
  1. Vyber premennú (napr. prvú prázdnu bunku),
  2. pre každú hodnotu z jej domény:
     - skontroluj `is_valid`,
     - priraď a rekurzívne rieš ďalej,
     - ak cesta vedie k riešeniu → hotovo,
     - inak zruš priradenie (backtrack) a skús ďalšiu hodnotu.
- **S MRV**:
  - krok (1) používa MRV namiesto „prvej prázdnej“ bunky.
- **S LCV**:
  - krok (2) mení poradie hodnôt podľa LCV.
- **S Forward Checking**:
  - po každom priradení pridáme kontrolu domén všetkých nepriradených premenných,
  - ak niektorá doména sa stane prázdnou → okamžitý backtrack.

### 6. Metrika expanded

- Rovnako ako v 8‑puzzle:
  - **expanded** = počet priradení (stavov), ktoré solver reálne skúsi (t. j. počet „uzlov“ v strome vyhľadávania),
  - slúži na porovnanie bruteforce, čistého backtrackingu, backtrackingu s MRV/LCV a forward checkingu.

---

## Zhrnutie druhej časti (od cvičenia 7)

- **Cvičenie 7**:
  - vyhľadávanie v stavovom priestore (8‑puzzle),
  - algoritmy BFS, DFS, Greedy best‑first, A*,
  - heuristiky (misplaced tiles, Manhattan distance),
  - pojmy: stav, operátor, cieľ, heuristika, admisibilita, konzistentnosť, úplnosť, optimálnosť, discovered/expanded.
- **Cvičenie 8**:
  - Sudoku ako CSP (premenné, domény, obmedzenia),
  - brute‑force DFS vs. backtracking,
  - heuristiky MRV a LCV,
  - Forward Checking a jeho efekt,
  - počet expandovaných stavov a čas riešenia ako metriky efektivity.
