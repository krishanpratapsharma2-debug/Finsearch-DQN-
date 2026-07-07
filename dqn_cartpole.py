"""
DQN for the Inverted Pendulum problem (CartPole-v1, Gymnasium)
================================================================
Mid-term report: Deep Reinforcement Learning for stock-trading optimisation
Algorithm demonstrated: Deep Q-Network (DQN, Mnih et al. 2013/2015)

Implemented FROM SCRATCH in NumPy — the Q-network, backpropagation and the
Adam optimiser are hand-coded, so every moving part of DQN is visible.

The inverted pendulum (CartPole) task: a pole is hinged on a cart that
slides on a frictionless track. The agent pushes the cart LEFT or RIGHT
each timestep to keep the pole upright. Reward = +1 per step survived.
Episode ends if |pole angle| > 12 deg, |cart position| > 2.4, or 500 steps.
CartPole-v1 is considered solved at average reward >= 475 over 100 episodes.

DQN components implemented here:
  1. Q-network        : MLP 4 -> 128 -> 128 -> 2 mapping state -> Q(s, a)
  2. Target network   : frozen copy, synced every TARGET_UPDATE learn steps
  3. Replay buffer    : stores transitions, sampled i.i.d. to break correlation
  4. Epsilon-greedy   : exploration rate decays from 1.0 to 0.01
  5. TD loss (Huber)  : (r + gamma * max_a' Q_target(s',a') - Q(s,a))^2

Usage:
  python dqn_cartpole.py [time_budget_seconds]   -> train (resumes checkpoint)
  python dqn_cartpole.py eval                    -> greedy-evaluate best_model.npz

The script checkpoints to dqn_checkpoint.npz and resumes automatically, so
training can be run in several short sessions. Whenever the greedy policy
achieves a new best evaluation score, its weights are saved to best_model.npz
(best-model snapshotting guards against DQN's catastrophic-forgetting dips).
"""

import os
import sys
import time

import gymnasium as gym
import numpy as np

# ----------------------------- hyper-parameters -----------------------------
SEED          = 42
GAMMA         = 0.99      # discount factor
LR            = 5e-4      # Adam learning rate
BATCH_SIZE    = 64        # minibatch sampled from the replay buffer
BUFFER_SIZE   = 50_000    # replay buffer capacity
MIN_BUFFER    = 1_000     # start learning after this many transitions
EPS_START     = 1.0       # initial exploration rate
EPS_END       = 0.01      # final exploration rate
EPS_DECAY     = 0.995     # multiplicative decay per episode
TARGET_UPDATE = 250       # sync target net every N gradient steps
MAX_EPISODES  = 1500
SOLVED_AVG    = 475.0     # solved threshold (mean of last 100 episodes)
HIDDEN        = 128
CKPT          = "dqn_checkpoint.npz"

rng = np.random.default_rng(SEED)


# ------------------------------- Q-network ---------------------------------
class QNetwork:
    """MLP 4 -> 128 -> 128 -> 2 with ReLU, hand-coded forward/backward + Adam."""

    def __init__(self, s_dim, n_act):
        def he(fan_in, shape):
            return rng.normal(0, np.sqrt(2.0 / fan_in), shape)
        self.p = {
            "W1": he(s_dim, (s_dim, HIDDEN)),  "b1": np.zeros(HIDDEN),
            "W2": he(HIDDEN, (HIDDEN, HIDDEN)), "b2": np.zeros(HIDDEN),
            "W3": he(HIDDEN, (HIDDEN, n_act)),  "b3": np.zeros(n_act),
        }
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        """x: (B, s_dim) -> Q: (B, n_act). Caches activations for backward."""
        self.x = x
        self.z1 = x @ self.p["W1"] + self.p["b1"]; self.a1 = np.maximum(self.z1, 0)
        self.z2 = self.a1 @ self.p["W2"] + self.p["b2"]; self.a2 = np.maximum(self.z2, 0)
        return self.a2 @ self.p["W3"] + self.p["b3"]

    def backward(self, dQ):
        """Backprop dLoss/dQ through the net; returns parameter gradients."""
        g = {}
        g["W3"] = self.a2.T @ dQ;              g["b3"] = dQ.sum(0)
        da2 = dQ @ self.p["W3"].T
        dz2 = da2 * (self.z2 > 0)
        g["W2"] = self.a1.T @ dz2;             g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.p["W2"].T
        dz1 = da1 * (self.z1 > 0)
        g["W1"] = self.x.T @ dz1;              g["b1"] = dz1.sum(0)
        return g

    def adam_step(self, g, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def copy_from(self, other):
        for k in self.p:
            self.p[k] = other.p[k].copy()


# ------------------------------ replay buffer ------------------------------
class ReplayBuffer:
    """Fixed-size circular buffer with uniform random minibatch sampling."""

    def __init__(self, capacity, s_dim):
        self.s  = np.zeros((capacity, s_dim), np.float32)
        self.a  = np.zeros(capacity, np.int64)
        self.r  = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, s_dim), np.float32)
        self.d  = np.zeros(capacity, np.float32)
        self.capacity, self.idx, self.size = capacity, 0, 0

    def push(self, s, a, r, s2, d):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        j = rng.integers(0, self.size, batch)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


# --------------------------------- agent -----------------------------------
class DQNAgent:
    def __init__(self, s_dim, n_act):
        self.n_act = n_act
        self.policy = QNetwork(s_dim, n_act)
        self.target = QNetwork(s_dim, n_act)
        self.target.copy_from(self.policy)
        self.buffer = ReplayBuffer(BUFFER_SIZE, s_dim)
        self.eps = EPS_START
        self.learn_steps = 0

    def act(self, state):
        """Epsilon-greedy: explore with prob eps, else greedy w.r.t. Q."""
        if rng.random() < self.eps:
            return int(rng.integers(self.n_act))
        q = self.policy.forward(state[None, :])
        return int(q.argmax())

    def learn(self):
        """One gradient step on the Huber TD loss over a sampled minibatch."""
        if self.buffer.size < MIN_BUFFER:
            return
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)

        # Bellman target: r + gamma * max_a' Q_target(s',a') * (1 - done)
        q_next = self.target.forward(s2).max(1)
        y = r + GAMMA * q_next * (1.0 - d)

        # Q(s,a) from the online network
        q_all = self.policy.forward(s)                     # (B, n_act)
        q_sa = q_all[np.arange(BATCH_SIZE), a]

        # Huber loss gradient w.r.t. Q(s,a): clip TD error to [-1, 1]
        td = q_sa - y
        dq_sa = np.clip(td, -1.0, 1.0) / BATCH_SIZE

        dQ = np.zeros_like(q_all)                          # grad only at taken action
        dQ[np.arange(BATCH_SIZE), a] = dq_sa
        self.policy.adam_step(self.policy.backward(dQ))

        self.learn_steps += 1
        if self.learn_steps % TARGET_UPDATE == 0:          # sync target network
            self.target.copy_from(self.policy)


# --------------------------- checkpoint / resume ---------------------------
def save_ckpt(agent, history, episode):
    np.savez_compressed(
        CKPT, episode=episode, eps=agent.eps, learn_steps=agent.learn_steps,
        history=np.array(history), t=agent.policy.t,
        buf_idx=agent.buffer.idx, buf_size=agent.buffer.size,
        bs=agent.buffer.s, ba=agent.buffer.a, br=agent.buffer.r,
        bs2=agent.buffer.s2, bd=agent.buffer.d,
        **{f"p_{k}": v for k, v in agent.policy.p.items()},
        **{f"m_{k}": v for k, v in agent.policy.m.items()},
        **{f"v_{k}": v for k, v in agent.policy.v.items()},
        **{f"tg_{k}": v for k, v in agent.target.p.items()},
    )


def load_ckpt(agent):
    z = np.load(CKPT)
    agent.eps = float(z["eps"]); agent.learn_steps = int(z["learn_steps"])
    agent.policy.t = int(z["t"])
    agent.buffer.idx, agent.buffer.size = int(z["buf_idx"]), int(z["buf_size"])
    agent.buffer.s, agent.buffer.a = z["bs"], z["ba"]
    agent.buffer.r, agent.buffer.s2, agent.buffer.d = z["br"], z["bs2"], z["bd"]
    for k in agent.policy.p:
        agent.policy.p[k] = z[f"p_{k}"]; agent.policy.m[k] = z[f"m_{k}"]
        agent.policy.v[k] = z[f"v_{k}"]; agent.target.p[k] = z[f"tg_{k}"]
    return int(z["episode"]), list(z["history"])


# ------------------------- greedy policy evaluation -------------------------
def greedy_eval(env, agent, n_episodes=5, seed_base=20_000):
    """Average return of the greedy (epsilon = 0) policy over fresh seeds."""
    total = []
    for k in range(n_episodes):
        s, _ = env.reset(seed=seed_base + k)
        done, ep_r = False, 0.0
        while not done:
            a = int(agent.policy.forward(s[None, :].astype(np.float32)).argmax())
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_r += r
        total.append(ep_r)
    return float(np.mean(total))


# -------------------------------- training ---------------------------------
def train(time_budget=None):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    start_ep, history = 0, []
    if os.path.exists(CKPT):
        start_ep, history = load_ckpt(agent)
        print(f"resumed at episode {start_ep}, eps={agent.eps:.3f}", flush=True)

    best_eval = -np.inf
    t0 = time.time()
    for episode in range(start_ep + 1, MAX_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        state = state.astype(np.float32)
        ep_reward, done = 0.0, False
        while not done:
            action = agent.act(state)
            s2, r, terminated, truncated, _ = env.step(action)
            s2 = s2.astype(np.float32)
            done = terminated or truncated
            # truncation (time limit) is not a true terminal state
            agent.buffer.push(state, action, r, s2, float(terminated))
            agent.learn()
            state = s2
            ep_reward += r

        agent.eps = max(EPS_END, agent.eps * EPS_DECAY)
        history.append(ep_reward)
        avg100 = float(np.mean(history[-100:]))

        if episode % 20 == 0:
            print(f"episode {episode:4d} | reward {ep_reward:6.1f} | "
                  f"avg100 {avg100:6.1f} | eps {agent.eps:.3f}", flush=True)

        # best-model snapshot: greedy-evaluate periodically once training warms up
        if episode % 10 == 0 and avg100 >= 200:
            score = greedy_eval(env, agent)
            if score > best_eval:
                best_eval = score
                np.savez("best_model.npz",
                         **{f"p_{k}": v for k, v in agent.policy.p.items()})

        if avg100 >= SOLVED_AVG and len(history) >= 100:
            print(f"SOLVED in {episode} episodes! avg100 = {avg100:.1f}", flush=True)
            save_ckpt(agent, history, episode)
            np.save("rewards_history.npy", np.array(history))
            env.close()
            return history, True

        if time_budget and time.time() - t0 > time_budget:
            save_ckpt(agent, history, episode)
            np.save("rewards_history.npy", np.array(history))
            print(f"time budget reached, checkpointed at episode {episode}", flush=True)
            env.close()
            return history, False

    save_ckpt(agent, history, MAX_EPISODES)
    np.save("rewards_history.npy", np.array(history))
    env.close()
    return history, False


def evaluate_best(n_episodes=20):
    """Greedy evaluation of best_model.npz on fresh, unseen seeds."""
    z = np.load("best_model.npz")
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    for k in agent.policy.p:
        agent.policy.p[k] = z[f"p_{k}"]
    scores = [greedy_eval(env, agent, 1, 30_000 + k) for k in range(n_episodes)]
    print(f"greedy eval over {n_episodes} unseen episodes: "
          f"mean {np.mean(scores):.1f} | min {min(scores):.0f} | max {max(scores):.0f}")
    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_best()
    else:
        budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
        history, solved = train(budget)
        print(f"episodes: {len(history)} | best: {max(history):.0f} | "
              f"avg100: {np.mean(history[-100:]):.1f} | solved: {solved}")
