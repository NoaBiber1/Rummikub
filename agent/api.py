from policies.greedy.rummikub_greedy import greedy_move

def get_move(hand, board, first_meld=False):
    return {"greedy_result": greedy_move(hand, board, first_meld), "max_q_result": ''}