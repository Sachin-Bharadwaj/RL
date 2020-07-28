import Gymwrappers as wrappers
import dqn_model
import argparse
import time
import  numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import pdb

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000 # maximum size
REPLAY_START_SIZE = 10_000 # size we wait before starting training
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000 # elapsed frames after which target network updated

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150_000

Experience = collections.namedtuple('Experience', \
                         field_names=['state', 'action', 'reward', 'done', \
                                       'new_state'])
class ExperienceBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, action, reward, dones, next_states = \
                 zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(action), np.array(reward, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

class Agent():
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad() # agent is not learning while playing, so disable grads
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        # choose action based on epsilon-greedy
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False) #?why list
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        # check if episode ended, then clear reward accumulator and reset env
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    #pdb.set_trace()
    indices = torch.arange(0,states.shape[0]).type(torch.long).to(device)
    state_action_values = net(states_v)[indices,actions_v.type(torch.long)]

    with torch.no_grad():
        next_state_action_values = tgt_net(next_states_v).max(1)[0]
        next_state_action_values[done_mask] = 0.0
        next_state_action_values = next_state_action_values.detach()

    expected_state_action_values = next_state_action_values * GAMMA + \
                                   rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        # decrement epsilon
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        # play one step
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"frame:{frame_idx}, games:{len(total_rewards)}, \
                  reward:{m_reward:.4f}, eps:{epsilon:.4f}, fps:{speed:.4f}")

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward

            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        # if Experience buffer less than threshold, then skip training
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Sync target network
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    writer.close()
