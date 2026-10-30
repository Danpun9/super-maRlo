import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros


class GymToGymnasiumWrapper(gym.Env):
    """
    Adapter wrapper that converts legacy OpenAI gym environment
    to the modern Farama Gymnasium API (5 return values for step, 2 for reset).
    """
    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

        # Convert observation space to Gymnasium space
        obs_shape = env.observation_space.shape
        obs_dtype = env.observation_space.dtype
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=obs_dtype
        )

        # Convert action space to Gymnasium space
        self.action_space = spaces.Discrete(env.action_space.n)

    def reset(self, **kwargs):
        # Gym legacy reset() returns only obs
        obs = self.env.reset()
        info = {}
        return obs, info

    def step(self, action):
        # Gym legacy step() returns obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="rgb_array"):
        try:
            return self.env.render(mode=mode)
        except Exception:
            return getattr(self.env.unwrapped, "screen", None)

    @property
    def screen(self):
        return getattr(self.env.unwrapped, "screen", None)

    def close(self):
        return self.env.close()


class SkipFrame(gym.Wrapper):
    """
    Skips consecutive frames (typically 4) and accumulates rewards.
    Takes max over the last 2 frames to deal with NES sprite flickering.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if i >= self._skip - 2:
                self._obs_buffer[i - (self._skip - 2)] = obs
            if term or trunc:
                terminated = term
                truncated = trunc
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info


class GrayScaleResizeObservation(gym.ObservationWrapper):
    """
    Converts RGB image to Grayscale (84x84) and sets channel-first shape (1, 84, 84).
    """
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = (1, self.shape[0], self.shape[1])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        # Convert RGB to Grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84 with bilinear interpolation
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        # Reshape to (1, 84, 84)
        return np.expand_dims(resized, axis=0)


def make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=True):
    """
    Factory function to create a fully wrapped Mario environment.
    """
    from envs.reward_wrappers import CustomMarioRewardWrapper

    # 1. Base legacy gym environment
    raw_env = gym_super_mario_bros.make(env_id)
    # 2. Joypad space mapping (7 actions)
    env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
    # 3. Gym to Gymnasium adapter
    env = GymToGymnasiumWrapper(env)
    # 4. Skip frames (4)
    env = SkipFrame(env, skip=4)
    # 5. Grayscale & resize to (1, 84, 84)
    env = GrayScaleResizeObservation(env, shape=84)
    # 6. Optional Custom Reward Wrapper
    if custom_reward:
        env = CustomMarioRewardWrapper(env)

    return env
