import torch
from  valid_solutions.ilp_solution import ILP_solutions

class GE:
    def __init__(self, players):
        self.players = players
        self.ilp_solver = ILP_solutions()
        self.reset()
        

    def play(self, x):
        action = x[-53:]
        act_sum = action.sum().item()

        if act_sum > 0:
            self.board += action
            self.hands[self.turn] -= action
        elif act_sum == 0:
            tail = self.draw(1)
            self.hands[self.turn][tail] += 1 
        else:
            raise ValueError(f"illegal action passed to play(): act_sum={act_sum} (must be >0 or ==0)")

        done_by_hand = self.hands[self.turn].sum().item()
        if done_by_hand == 0:
            self.done = True
            self.winner = self.turn
        if self.pointer >= self.deck.shape[0]:
            self.done = True

        self.turn =  (self.turn + 1) % self.players

    def is_Done(self):
        return self.done

    def hand_score(self, player):
        positions = torch.arange(1, 54, dtype=torch.float32)
        weights = torch.where(positions == 53, torch.tensor(30.0), ((positions - 1) % 13) + 1)
        return torch.sum(self.hands[player] * weights)

    def update_reward(self):
        if not self.done:
            return
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
        self.update_reward()
        return self.reward[player]

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