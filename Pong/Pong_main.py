import Gymwrappers as wrappers
import dqn_model
from PER import SumTree, Memory
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
MEAN_REWARD_BOUND = 17#19.0

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
class PrioReplayBuffer:
    def __init__(self,  buf_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity



    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        states, action, reward, dones, next_states = \
                 zip(*[self.buffer[idx] for idx in indices])
        samples  = tuple((np.array(states), np.array(action), \
                         np.array(reward, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)))
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


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
    def play_step(self, net, epsilon=0.0, noisy_dqn=False, device="cpu"):
        done_reward = None
        # choose action based on epsilon-greedy
        if noisy_dqn==False and  (np.random.random() < epsilon):
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False) #?why list -> adding Batch dimension
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

    @torch.no_grad() # agent is not learning while playing, so disable grads
    def play_n_step(self, net, epsilon=0.0, n_step=1, gamma=0.99, noisy_dqn=False, device="cpu"):
        done_reward = None
        n_step_reward = 0
        for i in range(n_step):
            # choose action based on epsilon-greedy
            if noisy_dqn==False and (np.random.random() < epsilon):
                action = self.env.action_space.sample()
            else:
                state_a = np.array([self.state], copy=False) #?why list -> adding Batch dimension
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())

            # cache first state, action in order to add to experience exp_buffer
            if i==0:
                init_state = self.state
                init_action = action

            # do step in the environment
            new_state, reward, is_done, _ = self.env.step(action)
            n_step_reward += gamma**(i-1) * reward # acc for n steps discounted reward
            self.total_reward += reward # global accumulator for undiscounted reward
            if is_done:
                break
            self.state = new_state

        exp = Experience(init_state, init_action, n_step_reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)

        # check if episode ended, then clear reward accumulator and reset env
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def calc_loss(batch, net, tgt_net, gamma=0.99, ddqn=False, batch_weights=None, \
              device="cpu"):

    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    if batch_weights is not None:
        batch_weights_v = torch.tensor(batch_weights).to(device)
    #pdb.set_trace()
    indices = torch.arange(0,states.shape[0]).type(torch.long).to(device)
    state_action_values = net(states_v)[indices,actions_v.type(torch.long)]

    with torch.no_grad():
        if ddqn:
            #pdb.set_trace()
            next_state_action = net(next_states_v).max(1)[1]
            next_state_action_values = tgt_net(next_states_v)[indices, next_state_action.type(torch.long)]
        else:
            next_state_action_values = tgt_net(next_states_v).max(1)[0]

        next_state_action_values[done_mask] = 0.0
        next_state_action_values = next_state_action_values.detach()

    expected_state_action_values = next_state_action_values * gamma + \
                                   rewards_v
    if batch_weights is None: # priority replay is disabled
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        priority = None
    else:
        loss = batch_weights_v * (state_action_values - \
                                  expected_state_action_values)**2
        priority = loss + 1e-5

        loss = loss.mean()

    return loss, priority

def calc_loss_n_steps(batch, net, tgt_net, gamma =0.99, n_steps=1, \
                      ddqn=False, batch_weights=None, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    if batch_weights is not None:
        batch_weights_v = torch.tensor(batch_weights).to(device)
    #pdb.set_trace()
    indices = torch.arange(0,states.shape[0]).type(torch.long).to(device)
    state_action_values = net(states_v)[indices, actions_v.type(torch.long)]

    with torch.no_grad():
        if ddqn:
            #pdb.set_trace()
            next_state_action = net(next_states_v).max(1)[1]
            next_state_action_values = tgt_net(next_states_v)[indices, next_state_action.type(torch.long)]
        else:
            next_state_action_values = tgt_net(next_states_v).max(1)[0]

        next_state_action_values[done_mask] = 0.0
        next_state_action_values = next_state_action_values.detach()

    expected_state_action_values = next_state_action_values * gamma**(n_steps) + \
                                   rewards_v

    if batch_weights is None: # priority replay is disabled
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        priority = None
    else:
        loss = batch_weights_v * (state_action_values - \
                                  expected_state_action_values)**2
        priority = loss + 1e-5

        loss = loss.mean()

    return loss, priority

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("--n_step", default=1, type=int, help="unrolling step in Bellman optimality eqn")
    parser.add_argument("--ddqn", default=0, help="set =1 to enable DDQN")
    parser.add_argument("--noisydqn", default=0, help='set to 1 to enable Noisy DQN/DDQN')
    parser.add_argument("--prioreplay", default=0, help='set to 1 to enable priority replay')
    parser.add_argument("--duelingdqn", default=0, help='set to 1 to enable priority replay')
    parser.add_argument("--BST_PER", default=0, help='set to 1 to enable Binary sum tree implementation of Experience Replay Memory')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = wrappers.make_env(args.env)
    if args.noisydqn:
        print("Choosing Noisy DQN architecture")
        net = dqn_model.NoisyDQN(env.observation_space.shape,
                        env.action_space.n).to(device)
        tgt_net = dqn_model.NoisyDQN(env.observation_space.shape,
                                env.action_space.n).to(device)
    elif args.duelingdqn:
        print("Choosing Dueling DQN architecture")
        net = dqn_model.DuelingDQN(env.observation_space.shape,
                        env.action_space.n).to(device)
        tgt_net = dqn_model.DuelingDQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    else:
        print("Choosing Vanilla DQN architecture")
        net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
        tgt_net = dqn_model.DQN(env.observation_space.shape,
                                env.action_space.n).to(device)

    if args.ddqn:
        print("Double DQN enabled")

    writer = SummaryWriter(comment="-" + args.env)
    print(net)


    if args.prioreplay:
        buffer = PrioReplayBuffer(REPLAY_SIZE)
        agent = Agent(env, buffer)
    elif args.BST_PER:
        buffer = Memory(REPLAY_SIZE)
        agent = Agent(env, buffer)
    else:
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
        if args.n_step == 1:
            reward = agent.play_step(net, epsilon, noisy_dqn=args.noisydqn, \
                                     device=device)
        else:
            reward = agent.play_n_step(net, epsilon=epsilon, \
                                       n_step=args.n_step, \
                                       gamma=GAMMA, noisy_dqn=args.noisydqn, \
                                       device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])


            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if args.noisydqn:
                for layer, snr in enumerate(net.noisylayer_snr()):
                    writer.add_scalar(f"layer:{layer}", snr , frame_idx)

                print(f"frame:{frame_idx}, games:{len(total_rewards)}, \
                      reward:{m_reward:.4f}, fps:{speed:.4f}")
            else:
                print(f"frame:{frame_idx}, games:{len(total_rewards)}, \
                      reward:{m_reward:.4f}, eps:{epsilon:.4f}, fps:{speed:.4f}")


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

        if args.prioreplay or args.BST_PER:
            batch, batch_indices, batch_weights = buffer.sample(BATCH_SIZE)
            #pdb.set_trace()
        else:
            batch = buffer.sample(BATCH_SIZE)
            batch_weights = None

        if args.n_step == 1:
            loss_t, priority = calc_loss(batch, net, tgt_net, gamma =GAMMA, \
                               ddqn=args.ddqn, batch_weights=batch_weights, \
                                device=device)
        else:
            loss_t, priority = calc_loss_n_steps(batch, net, tgt_net, gamma =GAMMA, \
                                     n_steps=args.n_step, ddqn=args.ddqn, \
                                     batch_weights=batch_weights, \
                                      device=device)
        loss_t.backward()
        optimizer.step()

        if args.prioreplay: # update priorities in buffer
            buffer.update_priorities(batch_indices, priority.data.cpu().numpy())

    writer.close()
