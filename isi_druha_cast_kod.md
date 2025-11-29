# Druhá časť – Dôležité časti kódu (od cvičenia 7)

Nižšie sú skrátené „skeletky“ kódu a vzory, ktoré je dôležité vedieť pochopiť/doplniť.

---

## Cvičenie 7 – 8‑puzzle a vyhľadávanie

### 1. Reprezentácia stavu – trieda `PuzzleState` (koncept)

```python
class PuzzleState:
    def __init__(self, board, parent=None, moved_tile=None):
        self.board = board          # 2D list, 0 = prázdne políčko
        self.size = len(board)
        self.parent = parent
        self.moved_tile = moved_tile
        self._blank_pos = self._find_blank()

    def _find_blank(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    return (r, c)

    def get_possible_moves(self):
        r, c = self._blank_pos
        moves = []
        # susedia prázdneho políčka
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.size and 0 <= cc < self.size:
                moves.append(self.board[rr][cc])
        return moves

    def move_tile(self, tile):
        """Vráti nový stav, kde je `tile` posunutý do prázdneho."""
        r, c = self._blank_pos
        new_board = [row[:] for row in self.board]

        # nájsť pozíciu dielika tile
        tr, tc = None, None
        for i in range(self.size):
            for j in range(self.size):
                if new_board[i][j] == tile:
                    tr, tc = i, j
                    break

        # prehodiť tile a 0
        new_board[r][c], new_board[tr][tc] = new_board[tr][tc], new_board[r][c]
        return PuzzleState(new_board, parent=self, moved_tile=tile)

    def is_goal(self):
        flat = [v for row in self.board for v in row]
        expected = list(range(1, self.size * self.size)) + [0]
        return flat == expected

    def reconstruct_path(self):
        path = []
        state = self
        while state.parent is not None:
            path.append(state.moved_tile)
            state = state.parent
        path.reverse()
        return path

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(tuple(tuple(r) for r in self.board))
```

### 2. Heuristiky pre PuzzleState

```python
    def count_misplaced_tiles(self):
        flat = [v for row in self.board for v in row]
        count = 0
        for i, val in enumerate(flat):
            if val == 0:
                continue
            if val != i + 1:
                count += 1
        return count

    def manhattan_distance(self):
        dist = 0
        for r in range(self.size):
            for c in range(self.size):
                val = self.board[r][c]
                if val == 0:
                    continue
                goal_r = (val - 1) // self.size
                goal_c = (val - 1) % self.size
                dist += abs(goal_r - r) + abs(goal_c - c)
        return dist
```

### 3. BFS – skeleton

```python
from collections import deque

def bfs(start_state):
    visited = set()
    queue = deque([start_state])

    discovered = 1
    expanded = 0

    while queue:
        state = queue.popleft()
        expanded += 1

        if state.is_goal():
            return state, discovered, expanded

        visited.add(state)

        for move in state.get_possible_moves():
            new_state = state.move_tile(move)
            if new_state not in visited and new_state not in queue:
                queue.append(new_state)
                discovered += 1

    return None, discovered, expanded
```

### 4. DFS – skeleton

```python
def dfs(start_state, limit=50_000):
    visited = set()
    stack = [start_state]

    discovered = 1
    expanded = 0

    while stack and len(visited) < limit:
        state = stack.pop()
        expanded += 1

        if state.is_goal():
            return state, discovered, expanded

        visited.add(state)

        # optional: reversed → kontroluje poradie detí
        for move in reversed(state.get_possible_moves()):
            new_state = state.move_tile(move)
            if new_state not in visited:
                stack.append(new_state)
                discovered += 1

    return None, discovered, expanded
```

### 5. Greedy best‑first search – skeleton

```python
import heapq, itertools

def greedy(start_state, heuristic_func):
    visited = set()
    heap = []
    counter = itertools.count()

    # (h(n), tie_break, state)
    heapq.heappush(heap, (heuristic_func(start_state), next(counter), start_state))
    discovered = 1
    expanded = 0

    while heap:
        _, _, state = heapq.heappop(heap)
        expanded += 1

        if state.is_goal():
            return state, discovered, expanded

        visited.add(state)

        for move in state.get_possible_moves():
            new_state = state.move_tile(move)
            if new_state not in visited:
                h = heuristic_func(new_state)
                heapq.heappush(heap, (h, next(counter), new_state))
                discovered += 1

    return None, discovered, expanded
```

### 6. A* (A‑star) – skeleton

```python
def astar(start_state, heuristic_func):
    visited = set()
    heap = []
    counter = itertools.count()

    g_score = {start_state: 0}
    f_score = {start_state: heuristic_func(start_state)}

    heapq.heappush(heap, (f_score[start_state], next(counter), start_state))

    discovered = 1
    expanded = 0

    while heap:
        _, _, state = heapq.heappop(heap)
        expanded += 1

        if state.is_goal():
            return state, discovered, expanded

        visited.add(state)

        for move in state.get_possible_moves():
            new_state = state.move_tile(move)
            tentative_g = g_score[state] + 1

            if new_state in visited and tentative_g >= g_score.get(new_state, float("inf")):
                continue

            if tentative_g < g_score.get(new_state, float("inf")) or new_state not in g_score:
                g_score[new_state] = tentative_g
                f_score[new_state] = tentative_g + heuristic_func(new_state)
                heapq.heappush(heap, (f_score[new_state], next(counter), new_state))
                discovered += 1

    return None, discovered, expanded
```

---

## Cvičenie 8 – Sudoku ako CSP

### 1. MRV – výber najviac obmedzenej premennej

```python
class SudokuSolver:
    def __init__(self, board, gui_callback=None):
        self.board = board
        self.expanded = 0
        self.gui_callback = gui_callback

    def select_unassigned_mrv(self):
        best_cell = None
        best_domain = None

        for r in range(9):
            for c in range(9):
                if self.board.grid[r][c] == 0:   # prázdna bunka
                    domain = self.board.get_domain(r, c)
                    if best_cell is None or len(domain) < len(best_domain):
                        best_cell = (r, c)
                        best_domain = domain

        return best_cell, best_domain
```

### 2. LCV – zoradenie hodnôt podľa toho, ako obmedzujú peer bunky

```python
    def sort_values_lcv(self, r, c, domain):
        peers = set()

        # riadok a stĺpec
        for i in range(9):
            if i != c:
                peers.add((r, i))
            if i != r:
                peers.add((i, c))

        # blok 3×3
        block_r = (r // 3) * 3
        block_c = (c // 3) * 3
        for rr in range(block_r, block_r + 3):
            for cc in range(block_c, block_c + 3):
                if (rr, cc) != (r, c):
                    peers.add((rr, cc))

        value_scores = {}

        for val in domain:
            score = 0
            for (rr, cc) in peers:
                if self.board.grid[rr][cc] != 0:
                    continue
                # ak peer nemôže prijať túto hodnotu, je „viac“ obmedzujúca
                if not self.board.is_valid(rr, cc, val):
                    score += 1
            value_scores[val] = score

        # hodnoty s menším score = najmenej obmedzujúce
        return sorted(domain, key=lambda v: value_scores[v])
```

### 3. DFS backtracking so zapínateľným MRV/LCV

```python
    def solve_dfs(self, use_mrv=False, use_lcv=False):
        self.expanded = 0
        return self._dfs(use_mrv, use_lcv)

    def _dfs(self, use_mrv, use_lcv):
        # výber bunky
        if use_mrv:
            cell, domain = self.select_unassigned_mrv()
            if cell is None:
                return True  # vyriešené
            r, c = cell
        else:
            pos = self.board.find_empty()
            if pos is None:
                return True
            r, c = pos
            domain = self.board.get_domain(r, c)

        if use_lcv:
            domain = self.sort_values_lcv(r, c, domain)

        for val in domain:
            if self.board.is_valid(r, c, val):
                self.board.grid[r][c] = val
                self.expanded += 1

                if self.gui_callback:
                    self.gui_callback()

                if self._dfs(use_mrv, use_lcv):
                    return True

                self.board.grid[r][c] = 0  # backtrack

        return False
```

### 4. Forward Checking – skeleton

```python
    def solve_forward_checking(self, use_mrv=False, use_lcv=False):
        self.expanded = 0
        return self._fc(use_mrv, use_lcv)

    def _fc(self, use_mrv, use_lcv):
        # výber bunky
        if use_mrv:
            cell, domain = self.select_unassigned_mrv()
            if cell is None:
                return True
            r, c = cell
        else:
            pos = self.board.find_empty()
            if pos is None:
                return True
            r, c = pos
            domain = self.board.get_domain(r, c)

        if use_lcv:
            domain = self.sort_values_lcv(r, c, domain)

        for val in domain:
            if self.board.is_valid(r, c, val):
                self.board.grid[r][c] = val
                self.expanded += 1

                if self.gui_callback:
                    self.gui_callback()

                # Forward Checking – kontrola domén všetkých nepriradených buniek
                all_domains_valid = True
                for rr in range(9):
                    for cc in range(9):
                        if self.board.grid[rr][cc] == 0:
                            if len(self.board.get_domain(rr, cc)) == 0:
                                all_domains_valid = False
                                break
                    if not all_domains_valid:
                        break

                if all_domains_valid:
                    if self._fc(use_mrv, use_lcv):
                        return True

                # undo
                self.board.grid[r][c] = 0

        return False
```

### 5. Brute‑force DFS (kontrast k CSP riešeniu)

```python
    def solve_bruteforce_dfs(self):
        self.expanded = 0
        return self._bruteforce_dfs(0, 0)

    def _bruteforce_dfs(self, r, c):
        if r == 9:
            return self._check_full_board()

        next_r = r + 1 if c == 8 else r
        next_c = 0 if c == 8 else c + 1

        if self.board.grid[r][c] != 0:
            return self._bruteforce_dfs(next_r, next_c)

        for val in range(1, 10):
            self.board.grid[r][c] = val
            self.expanded += 1

            if self._bruteforce_dfs(next_r, next_c):
                return True

        self.board.grid[r][c] = 0
        return False
```

Tieto skeletky sú typické miesta, kde môžeš mať na teste doplňovačky (napr. generovanie susedov, podmienka v A*, MRV výber bunky, forward checking kontrola domén a pod.).
