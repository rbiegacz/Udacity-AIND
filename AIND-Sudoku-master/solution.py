"""Python script solving Sudoku game"""

# pylint: disable = C0301
# pylint: disable = C0326
# pylint: disable = C0103
# pylint: disable = C0325

assignments = []

rows = 'ABCDEFGHI'
cols = '123456789'


def cross(a, b):
    "Cross product of elements in A and elements in B."
    return [s+t for s in a for t in b]

boxes = cross(rows, cols)

row_units = [cross(rr, cols) for rr in rows]
column_units = [cross(rows, cc) for cc in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
# Definining two additional units: diagonal unites
diagonal_units = [["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8", "I9"], ["I1", "H2", "G3", "F4", "E5", "D6", "C7", "B8", "A9"]]
# Adding diagonal units to a general list of units in Sudoku puzzle
unitlist = row_units + column_units + square_units + diagonal_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], []))- set([s])) for s in boxes)


def assign_value(vals, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if vals[box] == value:
        return vals

    vals[box] = value
    if len(value) == 1:
        assignments.append(vals.copy())
    return vals


def naked_twins(vals):
    """Eliminate values using the naked twins strategy.
    Args:
        vals(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers
    def remove_twins(v, u, b1, b2, b1_str):
        for bb in u:
            if bb == b1 or bb == b2:
                continue
            else:
                for s in b1_str:
                    assign_value(v, bb, v[bb].replace(s, ''))
        return v

    for unit in unitlist:
        for box1 in unit:
            if len(vals[box1]) != 2:
                continue
            for box2 in unit:
                if box1 == box2 or vals[box1] != vals[box2]:
                    continue
                else:
                    vals = remove_twins(vals, unit, box1, box2, vals[box1])
    return vals


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Input: A grid in string form.
    Output: A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))


def display(vals):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    if vals is None:
        print("Empty board. Is everything OK?")
        return
    width = 1+max(len(vals[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(vals[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF':
            print(line)
    return


def eliminate(vals):
    """
    Go through all the boxes, and whenever there is a box with a value, eliminate this value from the values of all its peers.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    solved_values = [box for box in vals.keys() if len(vals[box]) == 1]
    for box in solved_values:
        digit = vals[box]
        for peer in peers[box]:
            assign_value(vals, peer, vals[peer].replace(digit, ''))
    return vals


def only_choice(vals):
    """
    Go through all the units, and whenever there is a unit with a value that only fits in one box, assign the value to this box.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in vals[box]]
            if len(dplaces) == 1:
                assign_value(vals, dplaces[0], digit)
    return vals


def reduce_puzzle(vals):
    """
    Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
    If the sudoku is solved, return the sudoku.
    If after an iteration of both functions, the sudoku remains the same, return the sudoku.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    #solved_values = [box for box in vals.keys() if len(vals[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in vals.keys() if len(vals[box]) == 1])
        vals = eliminate(vals)
        vals = naked_twins(vals)
        vals = only_choice(vals)
        solved_values_after = len([box for box in vals.keys() if len(vals[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in vals.keys() if len(vals[box]) == 0]):
            return False
    return vals


def search(vals):
    "Using depth-first search and propagation, try all possible values."
    # First, reduce the puzzle using the previous function
    vals = reduce_puzzle(vals)
    if vals is False:
        return False ## Failed earlier
    if all(len(vals[s]) == 1 for s in boxes):
        return vals ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(vals[s]), s) for s in boxes if len(vals[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and
    for value in vals[s]:
        new_sudoku = vals.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    # let's convert a grid into values
    vals = grid_values(grid)

    # let's try to solve the Sudoku game
    vals = search(vals)
    # if return value is False or None then it means that we failed to solve Sudoku
    if vals is False or vals is None:
        return False
    else:
        return vals

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    values = solve(diag_sudoku_grid)

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)
    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
