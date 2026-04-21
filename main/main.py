#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-file reference implementation of
    Probabilistic Multi-Robot Planning with Temporal Tasks (DARS 2024)

    • LTL -> DRA  with ltl2dstar / ltl2ba      (no HOA)
    • DRA is parsed by promela.Parser          (DRA v2 explicit)&#8203;:contentReference[oaicite:0]{index=0}
    • Product-MDP  →  LP (prefix & suffix)     (PuLP + CBC solver)

Python deps (all pure-wheel):
    pip install networkx numpy pulp
Files that must sit next to this script:
    – ltl2dstar        (static build, chmod +x)
    – ltl2ba           (static build, chmod +x)
    – promela.py       (the parser you uploaded)
"""

from collections import defaultdict

import random
from subprocess import run, PIPE, CalledProcessError
from os.path import dirname, join, abspath
from collections import defaultdict
from itertools import product
import random, math, sys

import matplotlib.pyplot as plt
from matplotlib import colors

import networkx as nx
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatusOptimal

from promela import Parser            # 

# 1)  LTL  Deterministic Rabin Automaton  (DRA v2 explicit)
def run_ltl2dra(formula: str) -> str:
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
    return Parser(dra_text).parse()   # promela.Parser already checks consistency


def build_agent_formula(teams, base_task="F goal G ! risk"):
    teams = sorted(teams)

    # helper: prefix‑AND a list of sub‑formulas
    def conj(fs):
        return fs[0] if len(fs) == 1 else "& " + " ".join(fs)

    comm_part  = conj([f"G F comm_{t}" for t in teams])
    full_phi   = f"& {comm_part} {base_task}"
    return full_phi

class SubTeam:
    """
    Holds the rendez‑vous cells and current members for one logical team.
    Manages round-robin assignment of communication points so teams don’t get stuck.
    """
    def __init__(self, name, comm_cells):
        self.name         = name
        self.comm_cells   = set(comm_cells)   # all possible rendez-vous cells
        self.members      = []                # list[Agent]

        self.comm_list    = list(comm_cells)
        self.current_idx  = 0
        self.current_cell = self.comm_list[self.current_idx]

    def register(self, agent):
        self.members.append(agent)

    def next_comm(self):
        self.current_idx = (self.current_idx + 1) % len(self.comm_list)
        self.current_cell = self.comm_list[self.current_idx]
        return self.current_cell

# ---------- global registry -------------------------------------------------

SUB_TEAMS = {
    "T1": SubTeam("T1", {(1, 3), (6, 7)}),
    #"T1": SubTeam("T1", {(0, 5)}),
    "T2": SubTeam("T2", {(1, 1), (6, 6)}),
}

COMM_SETS = {name: team.comm_cells for name, team in SUB_TEAMS.items()}

class Scheduler:
    def __init__(self, sub_teams):
        self.sub_teams = sub_teams          # dict[str → SubTeam]
        self.present = {name: set() for name in sub_teams}

    # ---------- Manhattan helper --------------------------------------
    @staticmethod
    def _L1(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- public -------------------------------------------------

    def update(self, agent, t):

        for team_name in agent.teams:
            curr  = self.sub_teams[team_name].current_cell
            pres  = self.present[team_name]
            mems  = {m.id for m in self.sub_teams[team_name].members}
            poses = {m.id: m.state[0] for m in self.sub_teams[team_name].members}
            print(f"[t={t}] Team {team_name} @ {curr} | present={pres} | members at {poses}")

            cells = {self.sub_teams[team_name].current_cell}
            here  = agent.state[0] in cells

            if here:
                self.present[team_name].add(agent.id)
            else:
                self.present[team_name].discard(agent.id)

            # Check meeting condition
            full_team = {a.id for a in self.sub_teams[team_name].members}

            if self.present[team_name] == full_team:
                # require that *all* members are on the *same* grid cell
                locs = {ag.state[0] for ag in self.sub_teams[team_name].members}
                if len(locs) == 1:
                    meet_cell = locs.pop()
                    print(f"[t={t}] Team {team_name} met at {meet_cell}")
                    if team_name == "T1":
                        print("Team 1 met")
                        pass
                    if team_name == "T2":
                        print("Team 2 met")
                        #exit()
                        pass

                    self._new_round(team_name, t)

    # ------------------------------------------------------------------
    #  Choose next rendez‑vous & trigger replanning
    # ------------------------------------------------------------------
    def _new_round(self, team_name, t):
        team = self.sub_teams[team_name]

        new_cell = team.next_comm()
        print(f"[t={t}] Team {team_name} ➔ next rendez-vous = {new_cell}")

        for ag in team.members:
            ag.set_active_team(team_name)
            ag.replan_comm_cell(team_name, new_cell)

        self.present[team_name].clear()

class Agent:
    def __init__(self, id_, mdp, teams, base_task_phi, gamma=0.2):
        self.id    = id_            # integer agent index
        self.mdp   = mdp            # *its* MDP (may share grid parameters)
        self.teams = teams          # list/tuple of team labels

        self.base_task = base_task_phi

        self.active_team = teams[0]
        self.formula     = f"& G F comm_{self.active_team} {self.base_task}"

        self.gamma   = gamma          # risk tolerance

        self.product = None           # product MDP (filled by compile_policy)
        self.policy  = None           # dict[state→action]  (prefix+suffix merged)
        self.state   = None     # will be (w, q) after the compile policy


    def set_active_team(self, team_name):
        # rebuild formula to only care about the active team
        self.active_team = team_name
        self.formula = f"& G F comm_{team_name} {self.base_task}"
        self.compile_policy()

    def compile_policy(self):
        for t in self.teams:
            rc = SUB_TEAMS[t].current_cell
            self.mdp.C[t] = {rc}

        # if gamma==0, do pure-prefix only
        if self.gamma == 0.0:
            dra_txt = run_ltl2dra(self.formula)

            n, q0, edges, aps, rabin = parse_dra(dra_txt)

            # build the product MDP graph
            G, s0 = build_product(self.mdp, edges, aps, q0, n)

            dra_next = {q: {} for q in range(n)}
            for (q, q2), bitsets in edges.items():
                for bitmask in bitsets:
                    dra_next[q][bitmask] = q2

            rc = SUB_TEAMS[self.active_team].current_cell
            entry_states = {(w2, q2) for (w2, q2) in G.nodes() if w2 == rc}
            if not entry_states:
                raise RuntimeError(f"Agent {self.id}: cannot reach cell {rc}")

            π_pre = _occupation_lp(G, entry_states, 0.0, s0)

            self.policy    = π_pre
            self.ap_order  = aps
            self._dra_next = dra_next
            if self.state is None:
                self.state = s0
            return

        dra_txt = run_ltl2dra(self.formula)
        n, q0, edges, aps, rabin = parse_dra(dra_txt)

        G, s0 = build_product(self.mdp, edges, aps, q0, n)

        amec = amecs(G, [(p[0], p[1]) for p in rabin])
        if not amec:
            raise RuntimeError(f"Agent {self.id}: no accepting AMECs for this formula")

        filtered = []
        for C in amec:
            keep = True
            for team in self.teams:
                rc = SUB_TEAMS[team].current_cell
                # require at least one product-state (w,q) with w == rc
                if not any(w == rc for (w, _) in C):
                    keep = False
                    break
            if keep:
                filtered.append(C)

        print(f"Filtered AMECs for Agent {self.id} at rendez-vous { {team:SUB_TEAMS[team].current_cell for team in self.teams} } →\n    {filtered}")
        #import time
        #time.sleep(0.5)

        amec = filtered
        if not amec:
            raise RuntimeError(
                f"Agent {self.id}: no AMEC remains that visits the current rendez-vous"
            )

        self.product = G

        chosen_amec = min(amec, key=len)

        rc = SUB_TEAMS[self.active_team].current_cell
        entry_states = {
            (w2, q2)
            for (w2, q2) in G.nodes()
            if w2 == rc
        }

        if not entry_states:
            raise RuntimeError(
                f"Agent {self.id}: no entry state in AMEC for cell {SUB_TEAMS[self.teams[0]].current_cell}"
            )

        π_pre = _occupation_lp(G, entry_states, self.gamma, s0)

        π_suf = _occupation_lp(
            G.subgraph(chosen_amec).copy(),
            chosen_amec,
            0.0,                       # ensure reach with prob 1
            next(iter(chosen_amec))    # start anywhere in the AMEC
        )

        # merge prefix + suffix
        π_prod = π_pre
        π_prod.update(π_suf)

        self.policy    = π_prod          # key = (w,q)

        self.ap_order  = aps             # remember AP order once

        dra_next = {q: {} for q in range(n)}
        # `edges` is a dict:  (q,q2) -> {bitmask, bitmask, …}
        for (q, q2), bitset in edges.items():
            for bitmask in bitset:
                dra_next[q][bitmask] = q2
                self._dra_next = dra_next

        if self.state is None:
            self.state = s0                  # (w0, q0)

    # ---------- execution ----------
    def step(self):
        """Sample the next action from the stationary policy and move one step."""

        w, q = self.state

        for team in self.teams:
            if w == SUB_TEAMS[team].current_cell:
                return (self.state, "stay")

        # DEBUG
        #if w == (4, 5):
        #    dist = self.policy.get((w, q), {})
        #    print(f"[debug] at product (w={w},q={q}), π={(w,q)} → {dist}")
        #    import time
        #    time.sleep(1.5)
        #if w == (4, 6):
        #    dist = self.policy.get((w, q), {})
        #    print(f"[debug] at product (w={w},q={q}), π={(w,q)} → {dist}")
        #    import time
        #    time.sleep(1.5)

        dist = self.policy.get((w, q), {})
        if not dist:
            legal = [a for a in self.mdp.actions(w) if a != "stay"]

            a = random.choice(legal) if legal else "stay"

        elif isinstance(dist, str):
            a = dist
        elif dist:
            actions, probs = zip(*dist.items())
            a = random.choices(actions, weights=probs, k=1)[0]
        else:
            legal = [a for a in self.mdp.actions(w) if a != "stay"]
            a = random.choice(legal) if legal else "stay"

        w2 = self.mdp.next_state(w, a)
        

        bits = label_to_bits(self.mdp.label(w2), self.ap_order)

        q2   = self._dra_next[q][bits]

        self.state = (w2, q2)
        return self.state, a


    def at_comm_cell(self, team_name):
        """True iff the agent is currently on a rendez‑vous cell of `team`."""
        print("at_comm_cell", self.state[0], team_name)
        exit()
        return self.state[0] in self.mdp.C[team_name]

    def replan_comm_cell(self, team_name, new_cell):
        self.mdp.C[team_name] = {new_cell}
        self.compile_policy()


    def reset(self):
        self.state = self.mdp.start


# --------------------------------------------------------------------- #
# Minimal grid-world MDP used for demo
# --------------------------------------------------------------------- #
class GridRobotMDP:
    def __init__(self,
                 size=8,
                 p_fail=0.05,
                 obstacles=None,
                 comm_sets=None):            

        self.N   = size
        self.S   = [(x, y) for x in range(size) for y in range(size)
                    if not (obstacles and (x, y) in obstacles)]
        self.A   = ["N", "S", "E", "W", "stay"]
        self.pf  = p_fail
        self.start = (0, 0)

        self.C   = {k: set(v) for k, v in (comm_sets or {}).items()}

        self._delta = {
            "N": (0, 1),
            "S": (0, -1),
            "E": (1,  0),
            "W": (-1, 0),
            "stay": (0, 0),
        }

    def _legal(self, x, y):
        return 0 <= x < self.N and 0 <= y < self.N and (x, y) in self.S

    def next_state(self, s, a):

        if a not in self._delta:
            raise ValueError(f"unknown action {a}")

        if random.random() < self.pf:
            print("failure")
            return s                      # failure -> no motion

        dx, dy = self._delta[a]
        nx, ny = s[0] + dx, s[1] + dy
        return (nx, ny) if self._legal(nx, ny) else s

    # ---------------------------------------------------------------- #
    def states(self):           return self.S
    def actions(self, s):       return self.A

    def _risky_goal_flags(self, s):
        risky = (abs(s[0]-self.N//2) + abs(s[1]-self.N//2) <= 1)
        goal  = (s == (self.N-1, self.N-1))
        return {"risky": risky, "goal": goal}

    #def label(self, s):
    #    #print(s)
    #    return self._risky_goal_flags(s)  # dict[str->bool]
    def label(self, s):
        flags = self._risky_goal_flags(s)

        for team, cells in self.C.items():
            flags[f"comm_{team}"] = (s in cells)

        return flags

    def cost(self, s, a):
        # this guarantees that the robot will never linger on a goal state
        #if self.label(s)["goal"] and a == "stay":
        #    return 0.0          # free to linger on goal
        return 1.0

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
                        if (w2, q2) not in visited:
                            visited.add((w2, q2))
                            frontier.add((w2, q2))

                        if not G.has_node((w2, q2)):
                            G.add_node((w2, q2))

                        if "bits" not in G.nodes[(w2, q2)]:
                            G.add_node((w2, q2), bits=val)

                        G.add_edge(
                            (w, q), (w2, q2),
                            action=a,
                            prob=p_w,
                            cost=mdp.cost(w, a)
                        )
    return G, init_node

def amecs(G, rabin_pairs):
    amec_sets = []

    all_I = [I for (I, _) in rabin_pairs]
    q_sink = set.intersection(*[set(I) for I in all_I])

    for C in nx.strongly_connected_components(G):
        if not all(any(v in C for (_, v) in G.out_edges(u)) for u in C):
            continue

        ok = True
        for I, H in rabin_pairs:
            if not any(q in I for (_, q) in C) or any(q in H for (_, q) in C):
                ok = False
                break
        if not ok:
            continue

        if any(q in q_sink for (_, q) in C):
            continue

        amec_sets.append(set(C))
    return amec_sets

# --------------------------------------------------------------------- #
# Linear-programming  (prefix + suffix)
# --------------------------------------------------------------------- #

def reach_lp(G, target_set, s0, prob_req=1.0):
    edge_list = list(G.edges(data=True))               # fixed order
    y = [LpVariable(f"y_{i}", lowBound=0)
         for i in range(len(edge_list))]
    P = LpProblem("reach", LpMinimize)

    # ---------- objective ----------
    P += lpSum(y[i] * edge_list[i][2]["cost"] for i in range(len(edge_list)))

    # ---------- reach constraint ----------
    P += lpSum(
            y[i] * edge_list[i][2]["prob"]
            for i in range(len(edge_list))
            if edge_list[i][1] in target_set
        ) == prob_req                                   # == 1.0 for certainty

    # ---------- flow conservation ----------
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

    # ---------- stochastic policy ----------
    π = defaultdict(dict)
    for s in G.nodes():
        tot = sum(y[i].value()
                  for i in range(len(edge_list))
                  if edge_list[i][0] == s)
        if tot == 0:
            continue
        for i in range(len(edge_list)):
            u, v, d = edge_list[i]
            if u == s and y[i].value() > 0:
                π[s][d["action"]] = y[i].value() / tot

    init = {(edge_list[i][0], edge_list[i][2]["action"]): y[i].value()
            for i in range(len(edge_list))
            if edge_list[i][0] == s0 and y[i].value() > 0}

    return {"policy": π, "init_dist": init}


def _occupation_lp(G, goal, gamma, s0):
    P = LpProblem("stage", LpMinimize)

    edge_list = list(G.edges(data=True))              # fixed ordering
    y = [LpVariable(f"y_{i}", lowBound=0) for i in range(len(edge_list))]

    # ----- objective
    P += lpSum(y[i] * edge_list[i][2]["cost"] for i in range(len(edge_list)))

    # ----- reach-probability ≥ 1-gamma
    P += lpSum(
        y[i] * edge_list[i][2]["prob"]
        for i in range(len(edge_list))
        if edge_list[i][1] in goal
    ) >= 1 - gamma
    for s in G.nodes():
        inflow = lpSum(
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

    π = defaultdict(dict)
    for s in G.nodes():
        total = sum(
            y[i].value()
            for i in range(len(edge_list))
            if edge_list[i][0] == s
        )
        if total == 0:
            continue                              # state never visited
        for i in range(len(edge_list)):
            u, v, data = edge_list[i]
            if u != s:
                continue
            flow = y[i].value()
            if flow > 0:
                a = data["action"]
                π[s][a] = flow / total           # normalised probability

    return π


def prefix_suffix_policy(G, amec_sets, gamma, s0):
    goal = min(amec_sets, key=len)
    π_pre  = _occupation_lp(G, goal, gamma, s0)
    π_suf  = _occupation_lp(G.subgraph(goal).copy(), goal, 0.0, next(iter(goal)))
    π_pre.update(π_suf)

    return π_pre

def noisy_action(chosen, all_actions, eps=0.2):
    if random.random() > eps:
        return chosen
    alt = [a for a in all_actions if a != chosen]
    return random.choice(alt)

def run_mission(formula, grid_size=8, gamma=0.2, horizon=120,
                comm_points=None):                 

    dra_txt = run_ltl2dra(formula)
    n, q0, edges, aps, rabin = parse_dra(dra_txt)

    robot   = GridRobotMDP(size=grid_size,
                           comm_points=comm_points) 

    G, s0   = build_product(robot, edges, aps, q0, n)

    amec    = amecs(G, [(p[0], p[1]) for p in rabin])

    if not amec:  raise RuntimeError("No accepting AMECs – formula unsatisfiable?")

    π        = prefix_suffix_policy(G, amec, gamma, s0)

    cur = s0
    trace = []
    for _ in range(horizon):
        trace.append(cur[0])
        #a = π.get(cur, random.choice(robot.actions(cur[0])))
        if cur in π:
            actions, probs = zip(*π[cur].items())
            chosen = random.choices(actions, weights=probs, k=1)[0]
            print(chosen)
        else:
            # fallback if state never appeared in LP solution
            chosen = random.choice(robot.actions(cur[0]))
        a = noisy_action(chosen, robot.actions(cur[0]), eps=0.0)

        dist = robot.P(cur[0], a)
        rnd, cum = random.random(), 0.0
        for w2,p in dist.items():
            cum += p
            if rnd <= cum: break
        bits = label_to_bits(robot.label(w2), aps)
        # unique q2 (deterministic) given bits
        q2 = next(q2 for q2 in range(n)
                  if (cur[1],q2) in edges and bits in edges[(cur[1],q2)])
        cur = (w2, q2)
        print(cur)
        
    return trace

def visualize_grid(traj,                        # list[(x,y)]
                   grid_size=8,
                   goal_cell=None,             # (x,y) or None -> bottom-right
                   risky_cells=None,           # set[(x,y)]  or None -> cross
                   comm_cells=None,
                   ax=None,
                   show_plt=True):           # set[(x,y)] or None


    if goal_cell is None:
        goal_cell = (grid_size-1, grid_size-1)

    if risky_cells is None:
        c = grid_size // 2
        risky_cells = {(c, c), (c-1, c), (c+1, c), (c, c-1)}

    # ----- background colour map (0 = normal, 1 = risky, 2 = goal)
    if comm_cells is None:
        comm_cells = set()

    # ----- background colour map
    # 0 = normal, 1 = risky, 2 = goal, 3 = communication
    grid_vals = np.zeros((grid_size, grid_size))

    for (x, y) in risky_cells:
        grid_vals[y, x] = 1

    gx, gy = goal_cell
    grid_vals[gy, gx] = 2
    for (x, y) in comm_cells:
        grid_vals[y, x] = 3

    cmap   = colors.ListedColormap(
        ['white',        # 0 normal
         '#ffcccc',      # 1 risky   – light red
         '#c6ffc6',      # 2 goal    – light green
         '#fff2cc'])     # 3 comm    – light yellow
    bounds = [-.5, .5, 1.5, 2.5, 3.5]

    norm = colors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(grid_vals, cmap=cmap, norm=norm, origin='lower')

    # grid lines
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='k', linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    # trajectory
    xs, ys = zip(*traj)
    ax.plot(xs, ys, '-o', color='royalblue', linewidth=2, markersize=4)

    # annotations
    ax.text(gx, gy, 'G', va='center', ha='center', weight='bold')
    for (x, y) in risky_cells:
        ax.text(x, y, 'R', va='center', ha='center')

    for (x, y) in comm_cells:
        ax.text(x, y, 'C', va='center', ha='center')

    if show_plt:
        plt.title('Robot trajectory')
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------- #
# Tiny demo
# --------------------------------------------------------------------- #

if __name__ == "__main__":

    # ----- shared grid setup -------------------------------------------------
    risky = {(4, 4), (5, 2)}
    goal  = (7, 7)

    base_task = "& G ! risky F goal "

    # ----- build three agents on the same 8×8 grid -----------------------------
    mdp_template = dict(size=8, comm_sets=COMM_SETS)

    agents = [
        Agent(id_=0,
              mdp=GridRobotMDP(**mdp_template, obstacles=None),
              teams=["T1"],
              base_task_phi=base_task,
              gamma=0.0),
        Agent(id_=1,
              mdp=GridRobotMDP(**mdp_template, obstacles=None),
              teams=["T1", "T2"],
              base_task_phi=base_task,
              gamma=0.0),
        Agent(id_=2,
              mdp=GridRobotMDP(**mdp_template, obstacles=None),
              teams=["T2"],
              base_task_phi=base_task,
              gamma=0.0),
    ]

    # ----- build the scheduler -----------------------------------------------
    sched = Scheduler(SUB_TEAMS)

    for ag in agents:
        # tie the object back to its SubTeam(s)
        for t in ag.teams:
            SUB_TEAMS[t].register(ag)
        ag.compile_policy()

    # ----- run 30 time-steps --------------------------------------------------
    traj0 = [agents[1].state[0]]

    T = 500                      # simulate 60 steps so meetings can happen
    for t in range(1, T + 1):
        for ag in agents:
            ag.step()           # move once
            sched.update(ag, t) # tell scheduler where the agent is
        traj0.append(agents[1].state[0])   # keep plotting agent‑0

    visualize_grid(traj0,
                   grid_size=8,
                   goal_cell=goal,
                   risky_cells=risky,
                   comm_cells=set().union(*COMM_SETS.values())) 
