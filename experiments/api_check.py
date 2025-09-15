import gymnasium as gym
import numpy as np
from pettingzoo.mpe import simple_spread_v3

def run_gym_cartpole(episodes=3, max_steps=500, seed=42):
    print("\n=== Gymnasium: CartPole-v1 (random policy) ===")
    env = gym.make("CartPole-v1")
    ep_returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_r = 0.0
        for t in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        ep_returns.append(total_r)
        print(f"Episode {ep+1}: return={total_r:.1f}, steps={t+1}")
    env.close()
    print("Avg return:", np.mean(ep_returns))

def run_pettingzoo_mpe(episodes=2, seed=123):
    print("\n=== PettingZoo: MPE simple_spread_v3 (random policy) ===")
    # Headless default (no GUI). max_cycles defaults to 25; we'll set it explicitly.
    env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=50, continuous_actions=False)
    print("Agents:", env.possible_agents)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_rewards = {a: 0.0 for a in env.possible_agents}
        for agent in env.agent_iter():  # iterates until all agents done or max_cycles reached
            obs, reward, termination, truncation, info = env.last()
            action = None if (termination or truncation) else env.action_space(agent).sample()
            env.step(action)
            total_rewards[agent] = total_rewards.get(agent, 0.0) + reward
        print(f"Episode {ep+1}: returns={total_rewards}")
    env.close()

if __name__ == "__main__":
    run_gym_cartpole()
    run_pettingzoo_mpe()
    print("\nDay 1 check complete ")
