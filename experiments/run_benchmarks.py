"""
PRISM Benchmark Suite
=====================
Runs the Stackelberg controller and all baselines across AWS and Alibaba
traces, prints comparison tables, and generates publication-quality figures.

Usage:
    python -m experiments.run_benchmarks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Import from the actual core modules (no copy-paste) ──────────────────────
from core.solver import solve_stackelberg, leader_utility
from core.follower import follower_rate, mitigation

# ==============================================================================
# 1. STATEFUL QUEUEING ENVIRONMENT
# ==============================================================================

class StatefulRollingEnvironment:
    """Simulates physical edge device buffers, arrival traffic, and packet drops.

    Parameters
    ----------
    n_devices : int
        Number of IoT/CPS edge devices.
    max_queue : float
        Maximum buffer capacity per device (packets).
    theta, base_omega, r_bar, arrival_rate : np.ndarray or None
        Per-device parameters.  When None, use the 3-device defaults from
        the proposal's worked example.
    seed : int or None
        RNG seed for reproducible eviction draws.
    """

    def __init__(self, n_devices=3, max_queue=100.0, *,
                 theta=None, base_omega=None, r_bar=None,
                 arrival_rate=None, seed=None):
        self.n = n_devices
        self.Q = np.zeros(self.n)
        self.Q_max = np.full(self.n, max_queue)

        # Device & SLA parameters (configurable; defaults match the proposal)
        self.theta = theta if theta is not None else np.array([0.4, 0.3, 0.3])
        self.base_omega = base_omega if base_omega is not None else np.array([12.0, 13.0, 9.0])
        self.r_bar = r_bar if r_bar is not None else np.full(self.n, 10.0)
        self.arrival_rate = arrival_rate if arrival_rate is not None else np.array([4.0, 3.5, 5.0])

        # Counters
        self.total_dropped = 0.0
        self.total_generated = 0.0
        self.total_delivered = 0.0

        # Per-instance RNG for reproducibility
        self._rng = np.random.default_rng(seed)

    def get_w_t(self):
        """Buffer fullness scales device risk-sensitivity (panic multiplier)."""
        queue_fullness = self.Q / self.Q_max
        return self.base_omega * (1.0 + 3.0 * queue_fullness)

    def step(self, alpha, rho, eps_true, c_sp, c_od, c_r, D_tot,
             alpha_prev=0.0, delta=0.5):
        """Execute one time-step of the dynamic system.

        Returns
        -------
        u0 : float
            Leader utility this step.
        r_transmitted : np.ndarray
            Actual per-device transmission rates after eviction effects.
        dropped_step : float
            Total packets dropped (buffer overflow) this step.
        """
        w_t = self.get_w_t()
        eta_true = eps_true * mitigation(rho)

        r_transmitted = follower_rate(alpha, w_t, eta_true, self.r_bar)
        u0 = leader_utility(alpha, self.theta, w_t, eta_true, self.r_bar,
                            c_od, c_sp, c_r, rho, D_tot, alpha_prev, delta)

        # Eviction event realization (uses per-instance RNG)
        if self._rng.random() < eps_true:
            surviving_capacity = (1.0 - alpha) + (rho * alpha)
            r_transmitted = r_transmitted * surviving_capacity

        # Update physical device queues
        self.Q = self.Q + self.arrival_rate - r_transmitted
        overflows = np.maximum(0.0, self.Q - self.Q_max)
        dropped_step = float(np.sum(overflows))
        self.total_dropped += dropped_step

        self.total_generated += float(np.sum(self.arrival_rate))
        self.total_delivered += float(np.sum(r_transmitted))

        self.Q = np.clip(self.Q, 0.0, self.Q_max)
        return u0, r_transmitted, dropped_step


# ==============================================================================
# 2. TRACE LOADERS
# ==============================================================================

AWS_SPOT_ADVISOR_URL = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"
BRACKET_TO_EPS = {0: 0.025, 1: 0.055, 2: 0.090, 3: 0.135, 4: 0.185}


def load_aws_dataset(region="us-east-1", instance_type="c5.large",
                     c_od=10.0, T=720, seed=42):
    """Fetch the live AWS Spot Advisor feed and synthesise a price/risk trace."""
    import urllib.request
    import json

    print(f"-> Fetching live AWS Spot Advisor feed for {instance_type} ({region})...")
    try:
        req = urllib.request.Request(AWS_SPOT_ADVISOR_URL,
                                    headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        reg_data = data["spot_advisor"][region]["Linux"][instance_type]
        savings_pct = reg_data["s"]
        r_idx = reg_data["r"]
        base_c_sp = c_od * (1.0 - savings_pct / 100.0)
        base_eps = BRACKET_TO_EPS.get(r_idx, 0.08)
        print(f"   Success: Discount = {savings_pct}%, "
              f"Base Spot = ${base_c_sp:.2f}, Risk Bracket = {r_idx}")
    except Exception as e:
        print(f"   Notice: Web fetch fallback ({e}). Using calibrated defaults.")
        base_c_sp, base_eps = 3.90, 0.09

    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    diurnal = 0.45 * np.sin(2.0 * np.pi * t / 24.0)
    c_sp_trace = np.clip(
        base_c_sp + diurnal + rng.normal(0, 0.15, size=T), 1.0, c_od)
    eps_trace = np.clip(
        base_eps + 0.01 * diurnal + rng.normal(0, 0.005, size=T), 0.01, 0.25)
    return c_sp_trace, eps_trace


def load_alibaba_dataset(csv_path="data/alibaba_env_ready.csv",
                         default_T=2243, seed=42):
    """Load the Alibaba production cluster trace or synthesise a fallback."""
    if os.path.exists(csv_path):
        print(f"-> Ingesting Alibaba production trace: {csv_path}...")
        df = pd.read_csv(csv_path)
        return df["c_sp"].values, df["eps"].values

    print(f"-> Notice: '{csv_path}' not found. "
          f"Generating representative Alibaba-profile trace...")
    rng = np.random.default_rng(seed)
    t = np.arange(default_T, dtype=float)
    c_sp = np.clip(
        4.0 + 1.2 * np.sin(2.0 * np.pi * t / 48.0)
        + rng.normal(0, 0.35, size=default_T),
        1.5, 9.5)
    eps = np.clip(
        0.06 + 0.03 * (c_sp / 10.0) ** 2
        + rng.normal(0, 0.008, size=default_T),
        0.01, 0.25)
    return c_sp, eps


# ==============================================================================
# 3. POLICY BENCHMARK RUNNER
# ==============================================================================

def evaluate_policy(policy_name, c_sp_trace, eps_trace,
                    c_od=10.0, c_r=7.0, D_tot=80.0, delta=0.5, seed=42):
    """Run a single policy through the full trace and return aggregate metrics."""
    T = len(c_sp_trace)
    env = StatefulRollingEnvironment(n_devices=3, max_queue=100.0, seed=seed)

    alpha_prev = 0.0
    alpha_history = []
    rho_history = []
    u0_history = []
    cost_history = []
    queue_history = []

    reactive_cooldown = 0
    alpha_react = 0.8

    for t in range(T):
        c_sp_t = c_sp_trace[t]
        eps_t = eps_trace[t]
        w_t = env.get_w_t()

        if policy_name == "Stackelberg":
            rho_t = 0.30
            eta_t = eps_t * mitigation(rho_t)
            alpha_t, step_u0 = solve_stackelberg(
                env.theta, w_t, eta_t, env.r_bar, c_od, c_sp_t, c_r,
                rho_t, D_tot, alpha_prev=alpha_prev, delta=delta)

        elif policy_name == "All_OnDemand":
            alpha_t, rho_t = 0.0, 0.0
            eta_t = eps_t * mitigation(rho_t)
            step_u0 = leader_utility(
                alpha_t, env.theta, w_t, eta_t, env.r_bar, c_od, c_sp_t,
                c_r, rho_t, D_tot, alpha_prev, delta)

        elif policy_name == "All_Spot":
            alpha_t, rho_t = 1.0, 0.0
            eta_t = eps_t * mitigation(rho_t)
            step_u0 = leader_utility(
                alpha_t, env.theta, w_t, eta_t, env.r_bar, c_od, c_sp_t,
                c_r, rho_t, D_tot, alpha_prev, delta)

        elif policy_name == "Static_50":
            alpha_t, rho_t = 0.50, 0.20
            eta_t = eps_t * mitigation(rho_t)
            step_u0 = leader_utility(
                alpha_t, env.theta, w_t, eta_t, env.r_bar, c_od, c_sp_t,
                c_r, rho_t, D_tot, alpha_prev, delta)

        elif policy_name == "Reactive":
            rho_t = 0.10
            queue_pressure = np.max(env.Q / env.Q_max)
            if queue_pressure > 0.5 or reactive_cooldown > 0:
                alpha_t = 0.10
                reactive_cooldown = max(0, reactive_cooldown - 1)
                if queue_pressure > 0.8:
                    reactive_cooldown = 3
            else:
                alpha_t = min(0.90, alpha_react + 0.05)
            alpha_react = alpha_t
            eta_t = eps_t * mitigation(rho_t)
            step_u0 = leader_utility(
                alpha_t, env.theta, w_t, eta_t, env.r_bar, c_od, c_sp_t,
                c_r, rho_t, D_tot, alpha_prev, delta)

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        _, _, _ = env.step(alpha_t, rho_t, eps_t, c_sp_t, c_od, c_r,
                           D_tot, alpha_prev, delta)
        step_fleet_cost = ((1.0 - alpha_t) * c_od + alpha_t * c_sp_t
                           + rho_t * alpha_t * c_r)

        alpha_prev = alpha_t
        alpha_history.append(alpha_t)
        rho_history.append(rho_t)
        u0_history.append(step_u0)
        cost_history.append(step_fleet_cost)
        queue_history.append(float(np.mean(env.Q)))

    total_cost = float(np.sum(cost_history))
    total_u0 = float(np.sum(u0_history))
    drop_pct = (env.total_dropped / max(1e-9, env.total_generated)) * 100.0

    return {
        "policy": policy_name,
        "total_cost": total_cost,
        "total_u0": total_u0,
        "avg_spot": float(np.mean(alpha_history)) * 100.0,
        "dropped_packets": env.total_dropped,
        "drop_pct": drop_pct,
        "alpha_series": np.array(alpha_history),
        "u0_series": np.array(u0_history),
        "queue_series": np.array(queue_history),
    }


# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

POLICY_MARKERS = {
    "All_OnDemand": "o", "All_Spot": "s", "Static_50": "^",
    "Reactive": "D", "Stackelberg": "*",
}
POLICY_COLORS = {
    "All_OnDemand": "dimgray", "All_Spot": "crimson",
    "Static_50": "darkorange", "Reactive": "royalblue",
    "Stackelberg": "forestgreen",
}


def generate_visualizations(results_dict, c_sp_trace, eps_trace,
                            dataset_label="AWS"):
    """Generate and save Pareto and 3-panel trajectory plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\n-> Rendering plots for {dataset_label} evaluation...")

    # ── Plot 1: Pareto Trade-off ──────────────────────────────────────────
    plt.figure(figsize=(7.5, 4.8))
    for name, res in results_dict.items():
        s = 180 if name == "Stackelberg" else 90
        z = 5 if name == "Stackelberg" else 3
        plt.scatter(res["total_cost"], res["dropped_packets"],
                    marker=POLICY_MARKERS[name], color=POLICY_COLORS[name],
                    s=s, zorder=z, label=name)
        plt.annotate(
            f" {name}", (res["total_cost"], res["dropped_packets"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
            weight="bold" if name == "Stackelberg" else "normal")

    plt.xlabel("Total Fleet Infrastructure Cost ($)", fontsize=10)
    plt.ylabel("Cumulative Packet Drops (SLA Violations)", fontsize=10)
    plt.title(f"Pareto Trade-Off: Cost vs. SLA Violations ({dataset_label})",
              fontsize=11, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig_pareto_{dataset_label.lower()}.png")
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"   Saved: {path}")

    # ── Plot 2: 3-Panel Dynamic Trajectory ────────────────────────────────
    T_sub = min(200, len(c_sp_trace))
    hrs = np.arange(T_sub)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)

    # Panel A: Environment Volatility
    ax1 = axes[0]
    ax1.plot(hrs, c_sp_trace[:T_sub], color="steelblue", lw=1.6,
             label="Spot Price $c_{sp}(t)$")
    ax1.set_ylabel("Price ($)", color="steelblue")
    ax1_r = ax1.twinx()
    ax1_r.plot(hrs, eps_trace[:T_sub], color="crimson", lw=1.2, ls="--",
               label=r"Eviction Risk $\varepsilon(t)$")
    ax1_r.set_ylabel(r"Eviction Prob $\varepsilon$", color="crimson")
    ax1.set_title(
        f"Dynamic Controller Adaptation — First {T_sub} Hours ({dataset_label})",
        fontsize=11, weight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel B: Spot Allocation Decisions
    axes[1].plot(hrs, results_dict["Stackelberg"]["alpha_series"][:T_sub],
                 color="forestgreen", lw=2.0,
                 label=r"Proposed Stackelberg $\alpha^*(t)$")
    axes[1].plot(hrs, results_dict["Reactive"]["alpha_series"][:T_sub],
                 color="royalblue", lw=1.2, ls="-.",
                 label=r"Reactive Baseline $\alpha(t)$")
    axes[1].axhline(0.5, color="darkorange", ls=":", lw=1.2,
                    label="Static 50% Baseline")
    axes[1].set_ylabel(r"Spot Ratio $\alpha$", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=8.5, ncol=3)
    axes[1].grid(True, alpha=0.3)

    # Panel C: Cumulative Net Utility
    for name in ["Stackelberg", "Reactive", "All_OnDemand", "All_Spot"]:
        cum_u = np.cumsum(results_dict[name]["u0_series"][:T_sub])
        axes[2].plot(hrs, cum_u, label=name, color=POLICY_COLORS[name],
                     lw=1.8 if name == "Stackelberg" else 1.2)
    axes[2].set_xlabel("Elapsed Time (Hours)", fontsize=10)
    axes[2].set_ylabel(r"Net Utility $\sum U_0$", fontsize=10)
    axes[2].legend(loc="lower left", fontsize=8.5, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"fig_dynamic_adaptation_{dataset_label.lower()}.png")
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"   Saved: {path}")


# ==============================================================================
# 5. FORMATTED BENCHMARK REPORT
# ==============================================================================

def print_benchmark_table(title, results_dict):
    """Print a formatted comparison table to stdout."""
    base_od_cost = results_dict["All_OnDemand"]["total_cost"]

    print("\n" + "=" * 98)
    print(f" {title.upper()} ")
    print("=" * 98)
    header = (
        f"{'Policy':<15} | {'Fleet Cost':>11} | {'Savings ($)':>12} | "
        f"{'Savings (%)':>11} | {'Net Utility':>12} | {'Avg Spot %':>10} | "
        f"{'SLA Drops':>9}"
    )
    print(header)
    print("-" * 98)

    for name, res in results_dict.items():
        cost = res["total_cost"]
        savings_dlr = base_od_cost - cost
        savings_pct = ((savings_dlr / base_od_cost) * 100.0
                       if base_od_cost > 0 else 0.0)
        print(
            f"{name:<15} | ${cost:>10.2f} | ${savings_dlr:>11.2f} | "
            f"{savings_pct:>10.2f}% | ${res['total_u0']:>11.2f} | "
            f"{res['avg_spot']:>9.1f}% | {res['dropped_packets']:>9.0f}"
        )
    print("=" * 98)


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    policies = ["All_OnDemand", "All_Spot", "Static_50", "Reactive", "Stackelberg"]

    # Experiment 1: AWS Live EC2 Trace
    c_sp_aws, eps_aws = load_aws_dataset(
        region="us-east-1", instance_type="c5.large", T=720)
    aws_results = {p: evaluate_policy(p, c_sp_aws, eps_aws, seed=42)
                   for p in policies}
    print_benchmark_table(
        "Experiment 1: 30-Day AWS EC2 Trace Evaluation (720 Hours)",
        aws_results)
    generate_visualizations(aws_results, c_sp_aws, eps_aws, dataset_label="AWS")

    # Experiment 2: Alibaba Production Cluster Trace
    c_sp_ali, eps_ali = load_alibaba_dataset("data/alibaba_env_ready.csv")
    ali_results = {p: evaluate_policy(p, c_sp_ali, eps_ali, seed=42)
                   for p in policies}
    print_benchmark_table(
        "Experiment 2: Alibaba Cluster Workload Trace Evaluation",
        ali_results)
    generate_visualizations(ali_results, c_sp_ali, eps_ali,
                            dataset_label="Alibaba")

    print("\n[Done] All evaluations complete.")


if __name__ == "__main__":
    main()