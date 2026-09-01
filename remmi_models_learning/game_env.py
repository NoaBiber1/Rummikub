import torch
from  ilp_solution import ILP_solutions

class GE:
    def __init__(self, players, budget=None):
        if budget is None:
            raise ValueError(
                "GE requires an explicit budget dict "
                "({'max_actions', 'alt_counts', 'alts_per_count'}) - pass "
                "config['budget'] from the caller (see simulation.DEFAULTS)."
            )
        self.players = players
        self.ilp_solver = ILP_solutions(budget)
        self.reset()
        

    def play(self, x):
        action = x[-53:]
        act_sum = action.sum().item()

        # handle drop (update board and hand). NO intermediate reward: the
        # game pays out at the end, and the dense signal is PBRS, applied
        if act_sum > 0:
            self.board += action
            self.hands[self.turn] -= action

        #handle tile pull when no tiles in deck (end the game and update real score)
        elif self.pointer >= self.deck.shape[0] and act_sum == 0:
            self.done = True
            self.update_real_reward()

        #handle tile pull when there is tiles in deck (update hand)
        elif act_sum == 0:
            tail = self.draw(1)
            self.hands[self.turn][tail] += 1

        #raise an error when there is unvalid action
        else:
            raise ValueError(f"illegal action passed to play(): act_sum={act_sum} (must be >0 or ==0)")

        #check if the player end is tails to end the game
        done_by_hand = self.hands[self.turn].sum().item()
        if done_by_hand == 0:
            self.done = True
            self.winner = self.turn
            self.update_real_reward()

        self.turn =  (self.turn + 1) % self.players

    def is_Done(self):
        return self.done

    def hand_score(self, player):
        positions = torch.arange(1, 54, dtype=torch.float32)
        weights = torch.where(positions == 53, torch.tensor(30.0), ((positions - 1) % 13) + 1)
        return torch.sum(self.hands[player] * weights)

    def update_real_reward(self):
        # self.reward holds ONLY the terminal payoff - there is no accumulator
        # to subtract, so there is nothing here to alias. (A previous version
        # did `temp = self.reward` and later `self.reward -= temp`; tensor
        # assignment binds the same object, the in-place writes below mutated
        # `temp` too, and every reward came out exactly 0.)
        if self.winner != -1:
            total = 0.0
            for i in range(0, self.players):
                if self.winner != i:
                    self.reward[i] = - self.hand_score(i)
                    total -= self.reward[i]
            self.reward[self.winner] = total
        else: 
            scores = torch.stack([self.hand_score(i) for i in range(self.players)])
            winner_idx = torch.argmin(scores)
            winner_score = scores[winner_idx]
 
            total = 0.0
            for i in range(self.players):
                if i != winner_idx:
                    self.reward[i] = winner_score - scores[i]
                    total -= self.reward[i]
            self.reward[winner_idx] = total

    def get_reward(self, player):
        """The TRUE game reward for `player`: 0 until the game ends, then the
        zero-sum payoff. Indexed by the ARGUMENT, never by self.turn - turn has
        already advanced by the time a caller asks, and any caller asking about
        a seat that is not to move would silently get someone else's number.

        Deliberately unshaped, so history["episode_rewards"] and the
        vs-random / vs-greedy benchmarks keep measuring game outcomes. Shaping
        is added on top by the agent (simulation._play_episode)."""
        if not self.done:
            return torch.tensor(0.0)
        return self.reward[player] / 100

    def potential(self, player):
        """PHI(s) for potential-based reward shaping, from `player`'s seat.

        Score differential, on the same 1/100 scale as get_reward:
            PHI(s) = (sum of opponents' hand scores - own hand score) / 100
        At a hand-emptying terminal this is EXACTLY the true terminal reward
        for 2 players, which is what a good potential should be: an estimate of
        V*. It answers "how far ahead am I on the quantity that decides the
        game", not "have I dumped tiles" - the latter is the greedy heuristic
        and would bias learning towards the baseline the agent is benchmarked
        against.

        RETURNS 0.0 AT TERMINAL STATES, unconditionally. Ng et al. (1999)
        require PHI(terminal) = 0 for episodic tasks; without it the shaping
        does not telescope away and policy invariance is lost. Enforcing it
        here rather than at the call site means no caller can forget.

        (For >2 players the sum-of-opponents form is no longer exactly the
        terminal payoff for a loser. Every game here is 1-vs-1, but if that
        ever changes, revisit this.)"""
        if self.done:
            return 0.0
        own = self.hand_score(player)
        others = sum(self.hand_score(i) for i in range(self.players) if i != player)
        return float((others - own) / 100)

    def get_winner(self):
        if not self.done:
            return None
        if self.winner != -1:
            return self.winner
        scores = torch.stack([self.hand_score(i) for i in range(self.players)])
        return int(torch.argmin(scores).item())

    def shuffle(self):
        perm = torch.randperm(self.deck.shape[0])
        self.deck = self.deck[perm]
        self.pointer = 0

    def draw(self, n=1):
        if self.pointer + n > self.deck.shape[0]:
            raise RuntimeError("deck exhausted")

        tails = self.deck[self.pointer: self.pointer + n]
        self.pointer += n
        return tails

    def init_handes(self):
        tails = self.draw(self.players*14).view(self.players, 14)
        draws = torch.ones_like(tails, dtype=self.hands.dtype)
        self.hands.scatter_add_(1, tails, draws)

    def reset(self):
        self.board = torch.zeros(53)
        self.deck = torch.arange(53).repeat(2)
        self.hands = torch.zeros(self.players, 53)
        self.done = False
        self.reward = torch.zeros(self.players)
        self.turn = 0
        self.winner = -1
        self.shuffle()
        self.init_handes()

    def get_valid_x_list(self):
        hand = self.hands[self.turn]
        self.ilp_solver.reset(
            hand_tails=hand.round().int().tolist(),
            board_tails=self.board.round().int().tolist(),
        )
        flat = self.ilp_solver.build_action_set()
 
        valid_x_list = []
        for entry in flat:
            action = torch.tensor(entry["dropped_vector"], dtype=torch.float32)
            x = torch.cat([self.board, hand, action])
            valid_x_list.append(x)

        if self.pointer < self.deck.shape[0]:
            draw_x = torch.cat([self.board, hand, torch.zeros(53)])
            valid_x_list.append(draw_x)

        if not valid_x_list:
            self.done = True

        return valid_x_list