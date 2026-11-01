"""Game engine: Rummikub state, rules and the zero-sum terminal payoff."""

import torch
from  ilp_solution import ILP_solutions

class GE:
    """One game for `players` seats; every game in this project is 1-vs-1."""

    def __init__(self, players, budget=None):
        """Build the engine with an explicit ILP budget dict, and deal."""
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
        """Apply action x: place tiles when its sum is > 0, else draw one.

        Advances the turn and ends the game when a hand empties or the deck is
        exhausted. Raises ValueError on an illegal action.
        """
        action = x[-53:]
        act_sum = action.sum().item()

        if act_sum > 0:
            self.board += action
            self.hands[self.turn] -= action

        elif self.pointer >= self.deck.shape[0] and act_sum == 0:
            self.done = True
            self.update_real_reward()

        elif act_sum == 0:
            tail = self.draw(1)
            self.hands[self.turn][tail] += 1

        else:
            raise ValueError(f"illegal action passed to play(): act_sum={act_sum} (must be >0 or ==0)")

        done_by_hand = self.hands[self.turn].sum().item()
        if done_by_hand == 0:
            self.done = True
            self.winner = self.turn
            self.update_real_reward()

        self.turn =  (self.turn + 1) % self.players

    def is_Done(self):
        """True once the game has ended."""
        return self.done

    def hand_score(self, player):
        """Sum of `player`'s tile values (1-13 by position, joker 30)."""
        positions = torch.arange(1, 54, dtype=torch.float32)
        weights = torch.where(positions == 53, torch.tensor(30.0), ((positions - 1) % 13) + 1)
        return torch.sum(self.hands[player] * weights)

    def update_real_reward(self):
        """Write the zero-sum terminal payoff into self.reward."""
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
        """The TRUE reward for `player`.

        0 until the game ends, then the zero-sum payoff scaled by 1/100.

        Indexed by the ARGUMENT, never by self.turn, which has already advanced
        by the time a caller asks. Deliberately unshaped: shaping is added on
        top by the agent (simulation._play_episode).
        """
        if not self.done:
            return torch.tensor(0.0)
        return self.reward[player] / 100

    def potential(self, player):
        """PHI(s) for potential-based shaping, from `player`'s seat.

        (sum of opponents' hand scores - own hand score) / 100, on get_reward's
        scale, and 0.0 at any terminal state - which Ng et al. (1999) require
        for the shaping to telescope away. Exact only for 2 players.
        """
        if self.done:
            return 0.0
        own = self.hand_score(player)
        others = sum(self.hand_score(i) for i in range(self.players) if i != player)
        return float((others - own) / 100)

    def get_winner(self):
        """Winning seat index, or None while the game is unfinished."""
        if not self.done:
            return None
        if self.winner != -1:
            return self.winner
        scores = torch.stack([self.hand_score(i) for i in range(self.players)])
        return int(torch.argmin(scores).item())

    def shuffle(self):
        """Shuffle the deck and rewind the pointer."""
        perm = torch.randperm(self.deck.shape[0])
        self.deck = self.deck[perm]
        self.pointer = 0

    def draw(self, n=1):
        """The next n tiles; raises RuntimeError when the deck is empty."""
        if self.pointer + n > self.deck.shape[0]:
            raise RuntimeError("deck exhausted")

        tails = self.deck[self.pointer: self.pointer + n]
        self.pointer += n
        return tails

    def init_handes(self):
        """Deal 14 tiles to every seat."""
        tails = self.draw(self.players*14).view(self.players, 14)
        draws = torch.ones_like(tails, dtype=self.hands.dtype)
        self.hands.scatter_add_(1, tails, draws)

    def reset(self):
        """Start a fresh game: new board, deck, hands, turn and winner."""
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
        """Every legal x = [board | hand | action] for the player to move.

        Includes the all-zero draw action whenever the deck still holds tiles.
        An empty list means nothing is legal, and sets self.done.
        """
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