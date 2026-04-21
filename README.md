# Probabilistic Multi-Robot Planning with Temporal Tasks and Communication Constraints

This repository contains the reference implementation accompanying the paper:

> **Silva, T. C., Yu, X., & Hsieh, M. A. (2024). Probabilistic multi-robot planning with temporal tasks and communication constraints**. Auton Robot 50, 2 (2026). https://doi.org/10.1007/s10514-025-10231-6

The framework enables **probabilistic, formally grounded planning** for multi-robot teams operating under **stochastic motion**, **Linear Temporal Logic (LTL) task specifications**, and **intermittent communication constraints**. Robots coordinate through scheduled sub-team rendezvous points while computing local policies that provide **probabilistic guarantees** on task satisfaction.

---

## Overview

The system implements a **bi-level planning architecture**:

1. **High-level (offline):** A designer partitions the robot team into overlapping sub-teams, assigns communication schedules, and specifies local tasks and communication requirements as LTL formulas.
2. **Low-level (online/local):** Each robot translates its LTL formula into a Deterministic Rabin Automaton (DRA), constructs a product MDP with its motion model, identifies Accepting Maximal End Components (AMECs), and solves two coupled linear programs (prefix and suffix) to obtain a stochastic policy that satisfies the task with bounded risk γ.

During deployment, robots follow their policies, meet at rendezvous cells according to their schedules, share information, and locally re-plan with updated communication points.

---

## Repository Structure

```
.
├── main.py            # Full simulation: agents, scheduler, grid MDP, product MDP,
│                      #   LP solver, multi-agent execution loop, and visualization
├── ltl_tools.py       # Standalone single-robot pipeline (LTL → DRA → product → LP → simulate)
├── ltl2dra.py         # Thin wrapper: calls ltl2dstar and parses the output via promela.Parser
├── promela.py         # Parser for DRA v2 explicit format (output of ltl2dstar)
├── ltl2dstar          # Pre-compiled binary – LTL to Deterministic Rabin Automaton translator
├── ltl2ba             # Pre-compiled binary – LTL to Büchi Automaton (used internally by ltl2dstar)
└── README.md
```

## Quick Start

### Single-Robot Demo (`ltl_tools.py`)

Run a single robot on an 8×8 grid with the LTL formula "always eventually reach the goal, and always avoid risky cells":

```bash
python ltl_tools.py
```

### Multi-Robot Simulation (`main.py`)

Run three agents with two overlapping sub-teams (T1 and T2), each pursuing a local task while coordinating rendezvous:

```bash
python main.py
```

The default configuration:
- **3 agents** on an 8×8 grid with 5% actuation failure probability
- **Agent 0** belongs to T1; **Agent 1** belongs to T1 and T2; **Agent 2** belongs to T2
- **Communication task:** each agent must visit its sub-team's rendezvous cell infinitely often (`G F comm_Tk`)
- **Risk tolerance:** γ = 0.0 (deterministic reachability)
- **Simulation horizon:** 500 time steps

The simulation prints meeting events to stdout and displays a trajectory plot at the end.

---

## Rendezvous Selection Strategies

The framework supports pluggable rendezvous objective functions (see Section 3.3 of the paper). The current implementation uses **round-robin rotation** through a fixed set of candidate cells per sub-team. The paper additionally analyzes:

- **Min-Sum:** Minimize total Manhattan distance from all team members
- **Min-Max (Fairness):** Minimize the worst-case distance among team members
- **TSP-Style Patrol:** Cycle through communication cells in a pre-computed tour order
- **Risk-Weighted Sum:** Weight distances by remaining energy or local risk metrics
- **Hybrid:** Convex combination of Min-Max and Min-Sum with tunable parameter α

---

## Configuration

Key parameters can be adjusted in `main.py`:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `size` | `GridRobotMDP` | Grid dimension (default: 8×8) |
| `p_fail` | `GridRobotMDP` | Actuation failure probability (default: 0.05) |
| `obstacles` | `GridRobotMDP` | Set of blocked cells |
| `gamma` | `Agent` | Risk tolerance ∈ [0, 1]; 0 = must reach AMEC with certainty |
| `SUB_TEAMS` | module-level | Sub-team definitions with candidate rendezvous cells |
| `base_task` | `__main__` | LTL formula for the local task (prefix notation) |
| `T` | `__main__` | Simulation horizon (number of time steps) |

---

## Citation

If you use this code, please cite:

```bibtex
@article{Silva2026ProbabilisticMRP,
  author  = {Silva, Thales C. and Yu, Xi and Hsieh, M. Ani},
  title   = {Probabilistic multi-robot planning with temporal tasks and communication constraints},
  journal = {Autonomous Robots},
  volume  = {50},
  number  = {2},
  year    = {2026},
  doi     = {10.1007/s10514-025-10231-6}
}
```

---

## License

Please contact the authors for licensing information.
