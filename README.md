# PRISM: Game-Theoretic Cloud Cost Optimization for Cyber-Physical Systems

> **P**roactive **R**isk-aware **I**oT **S**tackelberg **M**odel for Elastic Spot Procurement

PRISM is a game-theoretic cloud resource controller for cyber-physical and edge-IoT networks. It models the interaction between cloud spot-instance pricing volatility and edge buffer constraints as a bilevel **Stackelberg leader-follower game**, optimizing spot procurement under SLA constraints without any ML training phase.

---

## Overview

Cloud Spot instances offer 60–90% cost discounts vs On-Demand compute but subject workloads to abrupt eviction notices (≈2 min on AWS). Reactive auto-scalers only shed Spot after delays occur, causing controller thrashing.

This project implements a **repeated Stackelberg game** with stateful migration friction and physical buffer queue dynamics:

- **Leader (Cloud Controller):** Selects optimal spot fraction α\* ∈ [0,1] and reserve ratio ρ\* to maximize net utility while mitigating eviction penalties and VM migration costs.
- **Followers (Edge Gateways):** Best-respond by adjusting transmission rates r_i\* based on announced risk η and physical queue backlog.

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

## Key Results

Benchmarked across 720 hours of live AWS EC2 market feeds and 2,243 steps of Alibaba production container traces:

### AWS EC2 30-Day Trace (`c5.large`, us-east-1)

| Policy | Fleet Cost | Savings vs OD | SLA Drops | Net Utility |
|:---|:---:|:---:|:---:|:---:|
| All On-Demand | $7,200 | — | 0 | -$5,474 |
| All Spot | $2,803 | 61.1% | 8,700 | -$7,988 |
| Reactive | $6,766 | 6.0% | 2,726 | -$5,758 |
| **Stackelberg** | **$6,700** | **6.9%** | **0** | **-$5,071** |

### Alibaba Production Trace

| Policy | Fleet Cost | Savings vs OD | SLA Drops | Net Utility |
|:---|:---:|:---:|:---:|:---:|
| All On-Demand | $22,430 | — | 0 | -$17,052 |
| Reactive | $19,604 | 12.6% | 0 | -$15,781 |
| **Stackelberg** | **$17,081** | **23.9%** | 3,156 | **-$15,233** |

---

## Project Structure

```
├── core/             # Stackelberg solver & follower best-response
├── simulator/        # Queue dynamics environment & trace loaders
├── baselines/        # Fixed-policy baselines (OD, Spot, Static, Reactive)
├── metrics/          # Cost savings & SLA violation calculators
├── experiments/      # Benchmark suite & figure generation
├── figures/          # Generated plots (Pareto, trajectory)
├── data/             # Trace data (Alibaba CSV)
├── main.py           # Single-run dynamic simulation
└── requirements.txt
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/shaazil/PRISM.git
cd PRISM

# Install dependencies
pip install -r requirements.txt

# Run the single dynamic trace loop
python main.py

# Run the full benchmark suite (AWS + Alibaba traces)
python -m experiments.run_benchmarks
```

---

## Citation

If you use this work, please cite the accompanying paper (in preparation).
