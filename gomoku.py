import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import pygame
import random
import tensorflow as tf
import datetime



class Gomoku(gym.Env):
    def __init__(self, size=19, render_mode="human", render_fps=1) -> None:
        super().__init__()
        self.size = size
        self.window_size = 512
        self.window = None
        self.clock = None
        self.canvas = None
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.observation_space = Box(low=-1, high=1, shape=(size, size), dtype=np.int8)
        self.board = np.zeros(shape=self.observation_space.shape)
        self.next_player = "blue"
        self.action_space = Discrete(self.size ** 2)
        # self.action_space = MultiDiscrete(nvec=(size, size))
        self.start_player = None

    def reset(self):
        self.board = np.zeros(shape=self.observation_space.shape)
        self.next_player = "blue"
        self.start_player = "blue"
        return np.zeros(shape=self.observation_space.shape), {"next player": self.next_player}

    def step(self, action):
        if self.next_player == "blue":
            self.next_player = "red"
            previous_player = "blue"
        else:
            self.next_player = "blue"
            previous_player = "red"
        assert self.action_space.contains(action), "invalid action"
        # Step 1: convert discrete action to position (i, j)
        i = action // self.size
        j = action - i * self.size
        # Step 2: check if position is occupied:
        if self.board[i, j] != 0:
            return None, -1000, True, False, {"start player": self.start_player,
                                              "winner": self.next_player}  # Next state, reward, done, truncated, info
        else:
            if previous_player == "blue":
                self.board[i, j] = 1
            else:
                self.board[i, j] = 2
            # Step 3: check if self.board result in a winning
            if self.win():
                return None, 1000, True, False, {"start player": self.start_player, "winner": previous_player}
            else:
                return self.board.copy(), 1, False, False, {}

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            pix_size = (self.window_size / self.size)
            for x in range(self.size + 1):
                pygame.draw.line(canvas, 0, (0, pix_size * x), (self.window_size, pix_size * x), width=3, )
                pygame.draw.line(canvas, 0, (pix_size * x, 0), (pix_size * x, self.window_size), width=3, )
            cells = np.nditer(self.board, flags=["multi_index"])
            for cell in cells:
                if cell == 1:
                    # Draw blue circle with position cells.multi_index
                    pygame.draw.circle(canvas, (0, 0, 255), (np.array(cells.multi_index) + 0.5) * pix_size,
                                       pix_size / 3, )
                elif cell == 2:
                    # Draw red circle with position cells.multi_index
                    pygame.draw.circle(canvas, (255, 0, 0), (np.array(cells.multi_index) + 0.5) * pix_size,
                                       pix_size / 3, )
                else:
                    pass
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def win(self):
        # check horizontal lines
        for row in self.board:
            consecutive_wins = 1
            for j in range(len(row)):
                if row[j] == row[j - 1] and row[j] != 0:
                    consecutive_wins += 1
                else:
                    consecutive_wins = 1
                if consecutive_wins == 5:
                    return True

        # check vertical lines
        for col in range(len(self.board[0])):
            consecutive_wins = 1
            for row in range(1, len(self.board)):
                if self.board[row][col] == self.board[row - 1][col] and self.board[row][col] != 0:
                    consecutive_wins += 1
                else:
                    consecutive_wins = 1
                if consecutive_wins == 5:
                    return True

        # check diagonal lines - top left -> bottom right
        for k in range(len(self.board) - 4):
            for l in range(4, len(self.board[0])):
                consecutive_wins = 1
                for m in range(5):
                    if self.board[k + m][l - m] == self.board[k][l] and self.board[k + m][l - m] != 0:
                        consecutive_wins += 1
                    else:
                        consecutive_wins = 1
                    if consecutive_wins == 5:
                        return True

        # check diagonal lines top right -> bottom left
        for t in range(len(self.board) - 4):
            for j in range(4, len(self.board[0])):
                consecutive_wins = 1
                for k in range(5):
                    if self.board[t + k][j - k] == self.board[t][j] and self.board[t + k][j - k] != 0:
                        consecutive_wins += 1
                    else:
                        break
                if consecutive_wins == 5:
                    return True
        return False


class RandomGomoku(Gomoku):
    def __init__(self) -> None:
        super().__init__()

    def reset(self):
        state, info = super().reset()
        if random.random() < 0.5:
            self.start_player = "blue"
            return state, info
        else:
            self.start_player = "red"
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            opponent_action = i * self.size + j
            opponent_next_state, opponent_reward, done, truncated, info = super().step(opponent_action)
            return opponent_next_state, info

    def step(self, action):
        next_state, reward, done, truncated, info = super().step(action)
        if done or truncated:
            return next_state, reward, done, truncated, info
        else:
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            opponent_action = i * self.size + j
            opponent_next_state, opponent_reward, opponent_done, opponent_truncated, info = super().step(opponent_action)
            if opponent_reward == 1:
                return opponent_next_state, reward, done, truncated, info
            else:
                return opponent_next_state, - opponent_reward, done, truncated, info


if __name__ == '__main__':

    env = RandomGomoku()
    obs = env.reset()
    done = False
    i = 0
    action_values = []

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train'
    writer = tf.summary.create_file_writer(train_log_dir)

    while not done:
        i += 1
        action = random.randint(0, 200)
        next_state, reward, done, truncated, info = env.step(action)
        with writer.as_default():
            tf.summary.scalar("reward", reward, step=i)
        writer.flush()
        print(info)
        env.render()
    print(i)
