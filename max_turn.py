import pulp
from itertools import combinations

# init game variables
COLORS = ['Red', 'Blue', 'Black', 'Orange']
NUMBERS = list(range(1, 14))
TILES = [(c, n) for c in COLORS for n in NUMBERS]

# generate all valid combinations of tiles
def generate_valid_sets():

    valid_sets = []
    # Groups - same number, different colors
    for n in NUMBERS:
        for length in [3, 4]:
            for combo in combinations(COLORS, length):
                valid_sets.append([(c, n) for c in combo])
    # Runs - same color, consecutive numbers
    for c in COLORS:
        for length in range(3, 14):
            for start in range(1, 15 - length):
                valid_sets.append([(c, n) for n in range(start, start + length)])
    return valid_sets

VALID_SETS = generate_valid_sets()

# find the optimal move for maximizing the number of tiles played from the hand
def solve_rummikub_turn(table_tiles, hand_tiles, table_jokers=0, hand_jokers=0):
   
    # c is counting the number of tiles on the table
    c = {t: table_tiles.count(t) for t in TILES}
    # h is counting the number of tiles in the hand
    h = {t: hand_tiles.count(t) for t in TILES}
    
    # init ILP problem
    prob = pulp.LpProblem("Rummikub_Solver_Exact", pulp.LpMaximize)
    
    # decision variables (include boundary constraints)
    # Xs: number of times each valid set will appear on the new board
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=2, cat='Integer') for i in range(len(VALID_SETS))]
    
    # Yt: number of tiles of type t that are played from the hand
    y = {t: pulp.LpVariable(f"y_{t[0]}_{t[1]}", lowBound=0, upBound=h[t], cat='Integer') for t in TILES}
    
    # Y_j: number of jokers that are played from the hand
    y_j = pulp.LpVariable("y_joker", lowBound=0, upBound=hand_jokers, cat='Integer')
    
    # Zt: number of jokers that are substituted for tiles of type t on the new board
    z = {t: pulp.LpVariable(f"z_{t[0]}_{t[1]}", lowBound=0, upBound=2, cat='Integer') for t in TILES}
    
    # objective function 
    # maximize the number of tiles played from the hand + the number of jokers played from the hand
    prob += pulp.lpSum([y[t] for t in TILES]) + y_j
    
    # constraints
    # 1. conservation of tiles (including jokers)
    for t in TILES:
        tiles_in_sets = pulp.lpSum([x[i] for i, v_set in enumerate(VALID_SETS) if t in v_set])
        prob += tiles_in_sets == c[t] + y[t] + z[t]
        
    # 2. conservation of jokers
    prob += pulp.lpSum([z[t] for t in TILES]) == table_jokers + y_j
    
    # solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # collect the results
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        return {"status": status, "message": "No valid move found"}
        
    chosen_sets = []
    for i, var in enumerate(x):
        for _ in range(int(var.varValue)):
            chosen_sets.append(VALID_SETS[i])
            
    joker_substitutions = []
    for t in TILES:
        if z[t].varValue and z[t].varValue > 0:
            for _ in range(int(z[t].varValue)):
                joker_substitutions.append(t)
            
    return {
        "status": status,
        "num_of_tiles_played": int(pulp.value(prob.objective)),
        "joker_acting_as": joker_substitutions,
        "new_board": chosen_sets
    }