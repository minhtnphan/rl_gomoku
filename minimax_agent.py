import random
from gomoku import Gomoku
import minimax_utils
import tensorflow as tf
import numpy as np



class MinimaxGomoku(Gomoku):

    def __init__(self) -> None:
        super().__init__()
        self.max_depth = 3
        self.best_move = None

    def step(self, action):
        row, col = action
        if self.next_player == "blue":
            self.next_player = "red"
            previous_player = "blue"
        else:
            self.next_player = "blue"
            previous_player = "red"

        if self.board[row][col] != 0:
            return None, -1000, True, False, {"start player": self.start_player,
                                              "winner": self.next_player}
        if previous_player == "blue":
            self.board[row][col] = 1
        else:
            self.board[row][col] = 2

        if self.win():
            return None, 1000, True, False, {"start player": self.start_player, "winner": previous_player}
        else:
            return self.board.copy(), 1, False, False, {}

    def minimax(self, depth=2):
        self.max_depth = depth
        if self.next_player == 'blue':
            score = self.minimax_helper(1, depth, -np.inf, np.inf)
        else:
            score = self.minimax_helper(2, depth, -np.inf, np.inf)

        row, col = self.best_move
        return score, row, col

    def minimax_helper(self, current_player, depth, alpha, beta):
        if depth <= 0:
            return minimax_utils.evaluate(self.board, current_player) - minimax_utils.evaluate(self.board,
                                                                                               3 - current_player)

        moves = minimax_utils.get_moves(self.board)
        best_move = None
        for score, r, c in moves:
            print(score, r, c)
            self.board[r][c] = current_player
            next_player = 3 - current_player
            score = -self.minimax_helper(next_player, depth - 1, -beta, -alpha)
            self.board[r][c] = 0
            if score > alpha:
                alpha = score
                best_move = (r, c)
                if alpha >= beta:
                    break

        if depth == self.max_depth and best_move:
            self.best_move = best_move

        return alpha


if __name__ == '__main__':
    env = MinimaxGomoku()
    obs = env.reset()
    done = False

    train_log_dir = 'logs/minimax'
    writer = tf.summary.create_file_writer(train_log_dir)

    i = 0
    action_values = []
    while not done:
        i += 1
        _, row, col = env.minimax()
        next_state, reward, done, truncated, info = env.step((row, col))
        with writer.as_default():
            tf.summary.scalar("reward", reward, step=i)
        writer.flush()
        print(info)
        env.render()
    print(i)
