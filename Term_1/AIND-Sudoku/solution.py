assignments = []
​
​
​
​
def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
​
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values
​
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values
​
def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
​
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # Find all instances of naked twins
    poss_twins = [box for box in boxes if len(values[box]) == 2]
    twins = [[box1, box2] for box1 in poss_twins for box2 in peers[box1]
            if values[box1] == values[box2]]
    # Eliminate the naked twins as possibilities for their peers
    for twin in twins:
        # Find all peers that the twins have in common and remove the twins values from the peers\
        shared_peers = [box for box in boxes if box in peers[twin[0]] and box in peers[twin[1]]]
        for peer in shared_peers:
            for val in values[twin[0]]:
                assign_value(values, peer, values[peer].replace(val, ""))
    return values
​
# Encode the board
def cross(a, b):
    "Cross product of elements in A and elements in B."
    return [s+t for s in a for t in b]
​
rows = "ABCDEFGHI"
cols = "123456789"
boxes = cross(rows,cols)
​
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
​
diag1 = [[str(r)+str(10- int(c)) for (r,c) in zip(rows,cols)]]
diag2 = [[r+c for (r,c) in zip(rows,cols)]]
​
# Build the list of units based off of whether or not we are solving the diagonal sudoku problem
diag_solve = 1
if diag_solve:
    unitlist = row_units + column_units + square_units + diag1 + diag2
else:
    unitlist = row_units + column_units + square_units
​
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)
​
def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = []
    all_digits = '123456789'
    for c in grid:
        if c == '.':
            values.append(all_digits)
        elif c in all_digits:
            values.append(c)
    assert len(values) == 81
    return dict(zip(boxes, values))
​
def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[s]) for s in boxes)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return 
​
def eliminate(values):
    """
    Use the elimination strategy to remove possible values for a box based off of its peers
    Args:
        values(dict): The sudoku puzzle in dictionary form
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
             assign_value(values, peer, values[peer].replace(digit,''))
    return values
​
def only_choice(values):
    """
    If there is only one choice for the box, assign it
    Args:
        values(dict): The sudoku puzzle in dictionary form
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                assign_value(values, dplaces[0], digit)
    return values
​
def reduce_puzzle(values):
    """
    Reduces a puzzle using the elimination and only choice strategies
    Args:
        values(dict): The sudoku puzzle in dictionary form
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Reduce the puzzle using elimination, only choice, and naked twins
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values
​
def search(values):
    """
    Using depth-first search and propagation, create a search tree and solve the sudoku.
    Args: 
        values(dict): The sudoku puzzle in dictionary form
    
    """
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False 
    if all(len(values[x]) == 1 for x in boxes):
        return values
    # Choose one of the unfilled squares with the fewest possibilities
    box, vals = sorted(values.items(), key = lambda x: 10 if (len(x[1]) <= 1) else len(x[1]))[0]
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in vals:
        new_sudoku = values.copy()
        assign_value(new_sudoku, box, value)
        attempt = search(new_sudoku)
        if attempt:
            return attempt
​
def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    values = search(values)
    
    return values
​
if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))
​
    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)
​
    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
