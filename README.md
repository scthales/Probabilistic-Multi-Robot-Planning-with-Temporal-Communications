# Probabilistic Multi-Robot Planning with Temporal Tasks and Communication Constraints

This repository contains code and examples accompanying the paper:

**Silva, T. C., Yu, X., & Hsieh, M. A. (2024). _Probabilistic Multi-Robot Planning with Temporal Tasks and Communication Constraints_.**  
(Extended AURO/DARS 2024 version) 

The work introduces a **probabilistic, formally grounded framework** for multi-robot planning under **stochastic motion**, **temporal logic tasks**, and **intermittent communication constraints**. Unlike classical planning approaches that assume persistent connectivity or deterministic task evolution, this method provides **probabilistic guarantees** on satisfaction of high-level temporal tasks while allowing teams to meet **only intermittently** via scheduled and flexible rendezvous points.

---

## Key Features

### Probabilistic Task Satisfaction  
Robots plan over **extended MDPs with probabilistic labels**, enabling planning under uncertainty in both motion and environment.

### LTL-Based Task Specification  
Local tasks and communication requirements are represented as **Linear Temporal Logic (LTL)** formulas, automatically translated to deterministic Rabin automata (DRAs).

### Bi-Level Planning  
- **High-level:** Team partitioning, communication schedules (sub-teams), and global logical structure.  
- **Low-level:** Each robot computes its own prefix/suffix policies locally.

### Intermittent Communication  
Robots only need to meet **infinitely often**, not continuously. A scheduling mechanism coordinates sub-team meetings without enforcing specific meeting times or locations.

### Flexible Rendezvous Selection  
Includes several rendezvous objective variations:
- Min-Sum distance  
- Min-Max (fairness)  
- Weighted risk-aware distances  
- TSP-style patrol  
- Hybrid objectives

### Linear Programming-Based Planner  
Prefix and suffix policies are derived by solving two linear programs:
- **Prefix LP:** maximize probability of reaching AMECs under risk γ.  
- **Suffix LP:** minimize steady-state cost within AMECs.

---

### Citation
If you use this code, please cite:
@article{silva2024probabilistic,
  title={Probabilistic Multi-Robot Planning with Temporal Tasks and Communication Constraints},
  author={Silva, Thales C and Yu, Xi and Hsieh, M. Ani},
  year={2024},
  journal={Autonomous Robots (AURO), Extended DARS 2024 Submission}
}


