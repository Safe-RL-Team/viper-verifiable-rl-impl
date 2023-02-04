import gym
import numpy as np
import pygame
from gym.spaces import Box, Discrete


# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
class ToyPong(gym.Env):
    width = 30
    height = 20
    min_speed = 1
    max_speed = 2
    # *half the paddle length
    paddle_length = 4
    # Must always be > max_speed to prevent ball from passing through paddle
    paddle_height = 2

    max_timesteps = 250

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    render_mode = None

    def __init__(self, args):
        if args.render:
            self.render_mode = "human"

        self.action_space = Discrete(3)
        # paddle_x, ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y
        self.observation_space = Box(low=np.array([0, 0, 0, self.min_speed, self.min_speed], dtype=np.float32),
                                     high=np.array(
                                         [self.width, self.width, self.height, self.max_speed, self.max_speed],
                                         dtype=np.float32))

        self.paddle_x = self.width / 2
        self.ball_pos_x = self.width / 2
        self.ball_pos_y = np.random.uniform(0, self.height / 2)
        self.ball_vel_x = self.rand_vel()
        self.ball_vel_y = self.rand_vel()
        self.t = 0
        self.window = None
        self.clock = None

    def rand_vel(self):
        return np.random.uniform(self.min_speed, self.max_speed) * np.random.choice([-1, 1])

    def reset(self):
        self.paddle_x = self.width / 2
        self.ball_pos_x = self.width / 2
        self.ball_pos_y = np.random.uniform(0, self.height / 2)
        self.ball_vel_x = self.rand_vel()
        self.ball_vel_y = self.rand_vel()
        self.t = 0
        self.window = None

        if self.render_mode == "human":
            self.render()

        return self.observation()

    def observation(self):
        return np.array([self.paddle_x, self.ball_pos_x, self.ball_pos_y, self.ball_vel_x, self.ball_vel_y])

    def step(self, action):
        assert self.action_space.contains(action)
        self.t += 1

        if action == 0:
            self.paddle_x = max(self.paddle_x - 1, 0)
        elif action == 1:
            self.paddle_x = min(self.paddle_x + 1, self.width)
        else:
            pass

        self.ball_pos_x += self.ball_vel_x
        self.ball_pos_y += self.ball_vel_y

        # Velocity is conserved
        if self.ball_pos_x < 0:
            self.ball_pos_x = 0
            self.ball_vel_x *= -1
        elif self.ball_pos_x > self.width:
            self.ball_pos_x = self.width
            self.ball_vel_x *= -1

        if self.ball_pos_y < 0:
            self.ball_pos_y = 0
            self.ball_vel_y *= -1

        done = False
        reward = 1
        # If the ball hits the paddle, it will bounce back
        # THe paddle needs a small implicit thickness (= ball vel y) so that the ball cannot pass through
        if (self.height + self.paddle_height >= self.ball_pos_y >= self.height) and \
                self.paddle_x - self.paddle_length <= self.ball_pos_x \
                <= self.paddle_x + self.paddle_length:
            self.ball_vel_y *= -1
        elif self.ball_pos_y > self.height:
            # import pdb; pdb.set_trace()
            done = True
            reward = 0

        if self.t >= self.max_timesteps:
            done = True

        if self.render_mode == "human":
            self.render()

        return self.observation(), reward, done, {}

    def render(self, mode="human"):
        scale = 10
        window_height = (self.height + self.paddle_height + 1) * scale

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width * scale, window_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width * scale, window_height))
        canvas.fill((255, 255, 255))

        # Player paddle
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (self.paddle_x - self.paddle_length) * scale,
                self.height * scale,
                (2 * self.paddle_length) * scale,
                self.paddle_height * scale,
            ),
        )

        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                self.ball_pos_x * scale - scale / 2,
                self.ball_pos_y * scale - scale / 2,
                scale,
                scale,
            ),
        )
        # Ball

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
