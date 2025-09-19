#!/usr/bin/env python3
"""
Day 2 — ε-greedy k‑armed bandit (stationary & non‑stationary)
Usage:
  python bandit_egreedy.py --steps 10000 --k 10 --eps 0.1 --nonstationary 1 --alpha 0.1 --seed 42 --out ../experiments/day02_bandit_runs.json
Notes:
- If alpha is provided (0<alpha<=1), we use a constant step-size (good for non-stationary).
- Else we use sample-average updates (1/N) (good for stationary).
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
import numpy as np

@dataclass
class KArmedBandit:
    k: int = 10
    sigma: float = 1.0
    nonstationary: bool = False
    rw_sigma: float = 0.01  # random-walk drift std for non-stationary case
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.q_true = self.rng.normal(0.0, 1.0, size=self.k)

    def step(self, a: int) -> float:
        r = self.rng.normal(self.q_true[a], self.sigma)
        if self.nonstationary:
            self.q_true += self.rng.normal(0.0, self.rw_sigma, size=self.k)
        return float(r)

def run_epsilon_greedy(steps=1000, k=10, eps=0.1, nonstationary=False, alpha=None, optimistic_init=None, seed=0):
    bandit = KArmedBandit(k=k, nonstationary=nonstationary, seed=seed)
    Q = np.zeros(k, dtype=float) if optimistic_init is None else np.ones(k)*optimistic_init
    N = np.zeros(k, dtype=int)
    rewards = np.zeros(steps, dtype=float)
    actions = np.zeros(steps, dtype=int)

    for t in range(steps):
        explore = bandit.rng.random() < eps
        a = bandit.rng.integers(k) if explore else int(np.argmax(Q))
        r = bandit.step(a)
        actions[t] = a
        rewards[t] = r
        N[a] += 1
        if alpha is None:
            Q[a] += (r - Q[a]) / N[a]
        else:
            Q[a] += alpha * (r - Q[a])
    return {
        "final_Q": Q.tolist(),
        "N": N.tolist(),
        "avg_reward": float(np.mean(rewards)),
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "nonstationary": nonstationary,
        "eps": eps,
        "alpha": alpha,
        "optimistic_init": optimistic_init,
        "k": k,
        "steps": steps,
        "seed": seed
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--nonstationary", type=int, default=0, help="1 for non-stationary, else 0")
    ap.add_argument("--alpha", type=float, default=None, help="constant step-size (recommended for non-stationary)")
    ap.add_argument("--optimistic_init", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = run_epsilon_greedy(
        steps=args.steps, k=args.k, eps=args.eps,
        nonstationary=bool(args.nonstationary),
        alpha=args.alpha, optimistic_init=args.optimistic_init,
        seed=args.seed
    )
    print(f"Average reward: {result['avg_reward']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
