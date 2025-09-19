#!/usr/bin/env python3
"""
Day 2 — Value Iteration on a tiny gridworld (default 4x4).
Usage:
  python value_iteration.py --n 4 --gamma 0.99 --theta 1e-8 --step_reward -1.0 \
      --out_v ../experiments/day02_V.npy --out_pi ../experiments/day02_PI.npy \
      --plot ../experiments/day02_vi_convergence.png
"""
from __future__ import annotations
import argparse, numpy as np
import matplotlib.pyplot as plt

ACTIONS = [(0,1),(1,0),(0,-1),(-1,0)]  # E,S,W,N

def build_gridworld(n=4, step_reward=-1.0, terminals=None, walls=None):
    if terminals is None:
        terminals = {(0,0): 0.0, (n-1,n-1): 0.0}
    walls = set() if walls is None else set(walls)
    S = [(i,j) for i in range(n) for j in range(n) if (i,j) not in walls]
    idx = {s:i for i,s in enumerate(S)}
    P = {}
    for s in S:
        if s in terminals:
            for a in range(4):
                P[(s,a)] = [(1.0, s, float(terminals[s]))]
            continue
        for a, (di, dj) in enumerate(ACTIONS):
            ni, nj = s[0] + di, s[1] + dj
            s_next = s
            if 0 <= ni < n and 0 <= nj < n and (ni,nj) not in walls:
                s_next = (ni, nj)
            r = step_reward
            P[(s,a)] = [(1.0, s_next, r)]
    return S, idx, P

def value_iteration(n=4, gamma=0.99, theta=1e-8, step_reward=-1.0, terminals=None, walls=None):
    S, idx, P = build_gridworld(n=n, step_reward=step_reward, terminals=terminals, walls=walls)
    V = np.zeros(len(S), dtype=float)
    deltas = []
    sweeps = 0
    while True:
        delta = 0.0
        for s in S:
            v_old = V[idx[s]]
            q_vals = []
            for a in range(4):
                q = sum(p * (r + gamma * V[idx[s_next]]) for p, s_next, r in P[(s,a)])
                q_vals.append(q)
            V[idx[s]] = max(q_vals)
            delta = max(delta, abs(v_old - V[idx[s]]))
        sweeps += 1
        deltas.append(delta)
        if delta < theta:
            break
    # greedy policy
    pi = np.zeros((len(S),), dtype=int)
    for s in S:
        q_vals = [sum(p*(r + gamma*V[idx[s_next]]) for p,s_next,r in P[(s,a)]) for a in range(4)]
        pi[idx[s]] = int(np.argmax(q_vals))
    return V, pi, deltas, S, idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--theta", type=float, default=1e-8)
    ap.add_argument("--step_reward", type=float, default=-1.0)
    ap.add_argument("--out_v", type=str, default=None)
    ap.add_argument("--out_pi", type=str, default=None)
    ap.add_argument("--plot", type=str, default=None)
    args = ap.parse_args()

    V, pi, deltas, S, idx = value_iteration(
        n=args.n, gamma=args.gamma, theta=args.theta, step_reward=args.step_reward
    )
    print(f"States: {len(S)} | Sweeps to converge: {len(deltas)} | Final delta: {deltas[-1]:.2e}")
    if args.out_v:
        np.save(args.out_v, V)
        print(f"Saved values to {args.out_v}")
    if args.out_pi:
        np.save(args.out_pi, pi)
        print(f"Saved policy to {args.out_pi}")
    if args.plot:
        plt.figure()
        plt.semilogy(deltas)
        plt.xlabel("Sweep")
        plt.ylabel(r"$\|V_{t+1}-V_t\|_\infty$")
        plt.title("Value Iteration Convergence")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print(f"Saved plot to {args.plot}")

if __name__ == "__main__":
    main()
