import gym
import time
import numpy as np
import argparse
import torch
import Gymwrappers as wrappers
import dqn_model
import collections

import pdb

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
FPS = 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest='vis',
                        help="Disable visualization",
                        action='store_false')
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n)

    state_dict = torch.load(args.model, map_location="cpu")
    net.load_state_dict(state_dict)
    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        #pdb.set_trace()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, info = env.step(action)
        if done:
            break

        if args.vis:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

    print(f"total reward: {total_reward}")
    print(f"Action counts: {c}")
    if args.record:
        env.env.close()
