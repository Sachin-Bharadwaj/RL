import gym
import cv2
import gym.spaces
import numpy as np
import collections

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import pdb

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip
        self._obs_buffer = collections.deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self._obs_buffer.append(obs)
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                 low=0, high=255, shape=(32,32,1),
                                 dtype= np.uint8)
    def observation(self, obs):
        return ProcessFrame84.process(obs)


    @staticmethod
    def process(frame):
        #pdb.set_trace()
        if frame.size == 240 * 256 * 3: # for SuperMarioBros
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert  False, "Unknown Resolution"
        #pdb.set_trace()
        # convert from 3 channel to grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # resize to 84,110 image (width, height)
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)

        img = np.reshape(img, (32,32,1)).astype(np.uint8)
        return img

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
                                 old_space.low.repeat(n_steps, axis=0),
                                 old_space.high.repeat(n_steps, axis=0),
                                 dtype = self.dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype= self.dtype)
        return self.observation(self.env.reset())

    def observation(self, obs):
        #pdb.set_trace()
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
                                 low=0.0, high=1.0, shape = new_shape,
                                 dtype = np.float32)

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return obs/255.0

def make_env(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    env = ScaledFloatFrame(env)
    #pdb.set_trace()
    return env
