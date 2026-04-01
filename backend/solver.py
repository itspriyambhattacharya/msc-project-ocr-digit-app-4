"""
backend/solver.py
=================
Sudoku solver using Backtracking + Minimum Remaining Values (MRV) heuristic.
MRV picks the empty cell with fewest valid candidates first,
drastically reducing the search tree vs naive left-to-right scan.
"""


# ── Constraint check ───────────────────────────────────────────────────────────

def _is_valid(board, idx, val):
    """Return True if placing val at board[idx] violates no sudoku rule."""
    row, col = divmod(idx, 9)

    # Row
    for c in range(9):
        if c != col and board[row*9+c] == val:
            return False
    # Column
    for r in range(9):
        if r != row and board[r*9+col] == val:
            return False
    # 3×3 box
    br, bc = (row // 3) * 3, (col // 3) * 3
    for dr in range(3):
        for dc in range(3):
            ni = (br+dr)*9 + (bc+dc)
            if ni != idx and board[ni] == val:
                return False
    return True


def _candidates(board, idx):
    """Return set of valid digits for an empty cell."""
    if board[idx] != 0:
        return set()
    return {n for n in range(1, 10) if _is_valid(board, idx, n)}


# ── MRV heuristic ─────────────────────────────────────────────────────────────

def _pick_cell(board):
    """
    MRV: choose the unfilled cell with the fewest legal candidates.
    Returns (index, candidate_set).
    If index == -1, all cells are filled (puzzle solved).
    """
    best_idx, best_n = -1, 10
    for i in range(81):
        if board[i] == 0:
            cands = _candidates(board, i)
            if len(cands) == 0:
                return i, set()        # dead end — trigger backtrack immediately
            if len(cands) < best_n:
                best_idx, best_n = i, len(cands)
    return best_idx, (_candidates(board, best_idx) if best_idx >= 0 else set())


# ── Backtracking solver ────────────────────────────────────────────────────────

def _backtrack(board):
    idx, cands = _pick_cell(board)
    if idx == -1:
        return True           # solved — all cells filled
    for n in sorted(cands):
        board[idx] = n
        if _backtrack(board):
            return True
        board[idx] = 0        # backtrack
    return False              # no candidate worked


# ── Conflict detector ─────────────────────────────────────────────────────────

def _find_conflicts(board):
    """
    Return deduplicated list of [i, j] pairs where board[i] == board[j]
    and they share a row, column, or box.
    """
    seen, conflicts = set(), []
    for i in range(81):
        v = board[i]
        if not v:
            continue
        row, col = divmod(i, 9)
        br, bc = (row//3)*3, (col//3)*3

        peers = (
            [row*9 + c for c in range(9) if c != col] +
            [r*9 + col for r in range(9) if r != row] +
            [(br+dr)*9 + (bc+dc)
             for dr in range(3) for dc in range(3)
             if (br+dr)*9+(bc+dc) != i]
        )
        for ni in peers:
            if board[ni] == v:
                key = (min(i, ni), max(i, ni))
                if key not in seen:
                    seen.add(key)
                    conflicts.append(list(key))
    return conflicts


# ── Public API ─────────────────────────────────────────────────────────────────

def solve_sudoku(flat_board):
    """
    Solve a sudoku puzzle.

    Parameters
    ----------
    flat_board : list[int]
        81 integers, 0 = blank, 1–9 = given digit.

    Returns
    -------
    dict with keys:
        solved    : bool
        board     : list[int]  — solution (or original if unsolvable)
        conflicts : list       — [[i,j], …] pairs of conflicting cells
        error     : str | None
    """
    conflicts = _find_conflicts(flat_board)
    if conflicts:
        return {
            "solved":    False,
            "board":     flat_board,
            "conflicts": conflicts,
            "error":     "Conflicting digits detected — see highlighted cells."
        }

    copy = flat_board[:]
    solved = _backtrack(copy)

    return {
        "solved":    solved,
        "board":     copy if solved else flat_board,
        "conflicts": [],
        "error":     None if solved else
        "No valid solution exists. Check for incorrect given digits."
    }
