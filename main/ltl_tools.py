#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subprocess import run, PIPE, CalledProcessError
from os.path import dirname, join, abspath
from collections import defaultdict
from itertools import product
import random, math, sys

import networkx as nx
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatusOptimal

from promela import Parser            # ← your DRA parser

def run_ltl2dra(formula: str) -> str:
    """Return the raw DRA text emitted by ltl2dstar in v2-explicit format."""
    here = dirname(__file__)
    ltl2dstar_bin = abspath(join(here, "ltl2dstar"))
    ltl2ba_bin    = abspath(join(here, "ltl2ba"))

    argv = [
        ltl2dstar_bin,
        f"--ltl2nba=spin:{ltl2ba_bin}",
        "--stutter=no",        # flip to 'yes' if you need stutter-invariance
        "-", "-"               # stdin / stdout
    ]
    try:
        res = run(argv, input=formula.encode(), stdout=PIPE, check=True)
    except CalledProcessError as e:
        raise RuntimeError(f"ltl2dstar failed (exit {e.returncode})") from None
    return res.stdout.decode()

def parse_dra(dra_text: str):
    """Return (n_states, q0, edges, ap_names, rabin_pairs) exactly as Parser does."""
    return Parser(dra_text).parse()   # promela.Parser already checks consistency

class GridRobotMDP:
    def __init__(self, size=8, p_fail=0.05, obstacles=None):
        self.N   = size
        self.S   = [(x, y) for x in range(size) for y in range(size)
                    if not (obstacles and (x, y) in obstacles)]
        self.A   = ["N", "S", "E", "W", "stay"]
        self.pf  = p_fail
        self.start = (0, 0)

    # ---------------------------------------------------------------- #
    def states(self):           return self.S
    def actions(self, s):       return self.A

    def _risky_goal_flags(self, s):
        risky = (abs(s[0]-self.N//2) + abs(s[1]-self.N//2) <= 1)
        goal  = (s == (self.N-1, self.N-1))
        return {"risky": risky, "goal": goal}

    def label(self, s):
        print(s)
        return self._risky_goal_flags(s)  # dict[str->bool]
    def cost(self, s, a):       return 1.0
    # stochastic motion ------------------------------------------------
    def P(self, s, a):
        if a == "stay":                 return {s: 1.0}
        dx, dy = {"N":(0,1),"S":(0,-1),"E":(1,0),"W":(-1,0)}[a]
        s2 = (s[0]+dx, s[1]+dy)
        if s2 not in self.S:            s2 = s
        return {s2: 1-self.pf, s: self.pf}

def label_to_bits(label_dict, ap_names) -> str:
    return ''.join(
        '1' if label_dict.get(ap, False) else '0'
        for ap in ap_names               # order from the parser!
    )


def build_product(mdp, dra_edges, ap_names, q0, n_states):
    G          = nx.DiGraph()
    init_node  = (mdp.start, q0)

    frontier   = {init_node}
    visited    = {init_node}
    G.add_node(init_node)

    while frontier:
        w, q = frontier.pop()

        for a in mdp.actions(w):
            for w2, p_w in mdp.P(w, a).items():
                val = label_to_bits(mdp.label(w2), ap_names)   # decimal str

                for q2 in range(n_states):
                    key = (q, q2)
                    if key in dra_edges and val in dra_edges[key]:
                        # enqueue BEFORE adding the edge
                        if (w2, q2) not in visited:
                            visited.add((w2, q2))
                            frontier.add((w2, q2))

                        G.add_edge(
                            (w, q), (w2, q2),
                            action=a,
                            prob=p_w,
                            cost=mdp.cost(w, a)
                        )
    return G, init_node

def amecs(G, rabin_pairs):
    amec_sets = []
    for C in nx.strongly_connected_components(G):
        for I, H in rabin_pairs:          # Parser puts I in index 0, H in index 1
            if any(q in I for _,q in C) and not any(q in H for _,q in C):
                if all(any(v in C for _,v in G.out_edges(u)) for u in C):
                    amec_sets.append(set(C))
    return amec_sets

def _occupation_lp(G, goal, gamma, s0):
    P = LpProblem("stage", LpMinimize)

    edge_list = list(G.edges(data=True))          # fixed ordering
    y = [LpVariable(f"y_{i}", lowBound=0) for i in range(len(edge_list))]

    P += lpSum(y[i] * edge_list[i][2]["cost"] for i in range(len(edge_list)))

    P += lpSum(
        y[i] * edge_list[i][2]["prob"]
        for i in range(len(edge_list))
        if edge_list[i][1] in goal
    ) >= 1 - gamma

    for s in G.nodes():
        inflow  = lpSum(
            y[i] * edge_list[i][2]["prob"]
            for i in range(len(edge_list))
            if edge_list[i][1] == s
        )
        outflow = lpSum(
            y[i]
            for i in range(len(edge_list))
            if edge_list[i][0] == s
        )
        P += outflow - inflow == (1 if s == s0 else 0)

    assert P.solve() == LpStatusOptimal

    π = {}
    for s in G.nodes():
        idxs = [i for i in range(len(edge_list)) if edge_list[i][0] == s]
        if not idxs:
            continue
        best_i = max(idxs, key=lambda i: y[i].value())
        if y[best_i].value() > 1e-9:
            π[s] = edge_list[best_i][2]["action"]
    return π


def prefix_suffix_policy(G, amec_sets, gamma, s0):
    goal = min(amec_sets, key=len)
    π_pre  = _occupation_lp(G, goal, gamma, s0)
    π_suf  = _occupation_lp(G.subgraph(goal).copy(), goal, 0.0, next(iter(goal)))
    π_pre.update(π_suf)
    return π_pre

def run_mission(formula, grid_size=8, gamma=0.2, horizon=120):
    dra_txt = run_ltl2dra(formula)
    n, q0, edges, aps, rabin = parse_dra(dra_txt)

    robot   = GridRobotMDP(size=grid_size)

    G, s0   = build_product(robot, edges, aps, q0, n)

    amec    = amecs(G, [(p[0], p[1]) for p in rabin])

    if not amec:  raise RuntimeError("No accepting AMECs – formula unsatisfiable?")
    π        = prefix_suffix_policy(G, amec, gamma, s0)

    cur = s0
    trace = []
    for _ in range(horizon):
        trace.append(cur[0])
        a = π.get(cur, random.choice(robot.actions(cur[0])))
        dist = robot.P(cur[0], a)
        rnd, cum = random.random(), 0.0
        for w2,p in dist.items():
            cum += p
            if rnd <= cum: break
        bits = label_to_bits(robot.label(w2), aps)
        q2 = next(q2 for q2 in range(n)
                  if (cur[1],q2) in edges and bits in edges[(cur[1],q2)])
        cur = (w2, q2)
    return trace

if __name__ == "__main__":
    phi =  "& G F goal G ! risky"
    traj = run_mission(phi, grid_size=8, gamma=0.3, horizon=60)
    print("first 15 workspace states:", traj[:15])
