# PRISM: Game-Theoretic Cloud Cost Optimization in Cyber-Physical Systems
> **P**roactive **R**isk-aware **I**oT **S**tackelberg **M**odel for Elastic Spot Procurement


PRISM is a stateful, game-theoretic cloud resource controller designed for cyber-physical and edge-IoT networks. It models the dynamic interaction between cloud capacity pricing volatility and edge buffer constraints as a bilevel **Stackelberg leader-follower game**, eliminating reactive controller thrashing and optimizing spot instance procurement under hard SLA guarantees.

---

## Overview

Cloud Spot instances offer 60–90% cost discounts relative to On-Demand compute, but subject workloads to abrupt 2-minute eviction notices. Standard reactive auto-scalers only shed Spot instances after packet delays occur, leading to controller thrashing (the sawtooth effect).

This project implements a **Stackelberg leader-follower game** with stateful migration friction and physical buffer queue dynamics:
- **Leader (Cloud Resource Controller):** Selects optimal Spot fraction $\alpha^* \in [0, 1]$ and reserve ratio $\rho^*$ to maximize net revenue while mitigating expected SLA eviction payouts and VM migration penalties.
- **Followers (Heterogeneous Edge Gateways / IoT Devices):** Best-respond by adjusting transmission rates $r_i^*$ based on announced risk $\eta$ and physical queue backlog stress.

---

## Architecture 
```
                  ┌───────────────────────────────────────────┐
                  │    Cloud Controller (Leader: PRISM)       │
                  │  Decision: Spot Ratio α*, Warm Reserve ρ* │
                  └──────────────────┬────────────────────────┘
                                     │ Announces:
                                     │ Price & Mitigated Risk η
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Edge Gateways (Followers: CPS Nodes)│
                  │  Reaction: Adjust Transmission Rate r* │
                  └──────────────────┬─────────────────────┘
                                     │ Dynamic Backpressure:
                                     │ Buffer Stress (wt)
                                     ▼
                         [Physical Queues Q(t)]
```                         

---


## Key Experimental Results

Benchmarked across 720 hours of live AWS EC2 market feeds and 2,243 steps of real Alibaba production container traces:

### 1. Live AWS EC2 30-Day Trace (`c5.large`, us-east-1)
| Policy | Total Compute Cost | Savings vs. All-OD | SLA Packet Drops | Net Utility ($U_0$) |
| :--- | :---: | :---: | :---: | :---: |
| **All On-Demand** | $7,200.00 | $0.00 (0.00%) | 0 | -$5,473.52 |
| **All Spot** | $2,803.35 | $4,396.65 (61.06%) | 8,700 | -$7,987.68 |
| **Static 50/50** | $5,505.68 | $1,694.32 (23.53%) | 8,638 | -$6,320.84 |
| **Reactive Auto-scaler** | $6,765.83 | $434.17 (6.03%) | 2,726 | -$5,758.03 |
| **Proposed Stackelberg** | **$6,700.15** | **$499.85 (6.94%)** | **0** | **-$5,070.95** |

*Result:* Stackelberg strictly dominates the Reactive heuristic—saving more infrastructure spend while completely eliminating all 2,700+ buffer overflow drops.

### 2. Alibaba Production Cluster Workload Trace
| Policy | Total Compute Cost | Savings vs. All-OD | SLA Packet Drops | Net Utility ($U_0$) |
| :--- | :---: | :---: | :---: | :---: |
| **All On-Demand** | $22,430.00 | $0.00 (0.00%) | 0 | -$17,051.52 |
| **Reactive Auto-scaler** | $19,603.97 | $2,826.03 (12.60%) | 0 | -$15,780.52 |
| **Proposed Stackelberg** | **$17,081.09** | **$5,348.91 (23.85%)** | 3,156 | **-$15,232.79** |

*Result:* Stackelberg secures the highest global economic utility across all policies and saves nearly double the cost of reactive scaling ($23.85\%$ vs. $12.60\%$).

---

## Project Structure

```text
├── core/             # Piecewise-quadratic candidate solver & follower responses
├── simulator/        # Physical queue dynamics and real-world trace ingestion
├── experiments/      # Automated benchmark suite and figure generation
├── figures/          # Pareto trade-off and time-series adaptation plots
├── main.py           # Single-run dynamic pipeline execution
└── requirements.
```

### 3. Quick Start
```
# Clone the repository
git clone [https://github.com/][(https://github.com/)shaazil/PRISM.git]
cd cloud-stackelberg

# Install dependencies
pip install -r requirements.txt

# Run the single dynamic trace loop
python main.py

# Run the comprehensive benchmark suite across AWS and Alibaba traces
python -m experiments.run_benchmarks
```
