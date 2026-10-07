import numpy as np
from simulator.trace_loader import fetch_web_spot_trace
from simulator.rolling_env import RollingEnvironment
from core.solver import solve_stackelberg, leader_utility

def run_policy(policy_name, c_sp_trace, eps_trace, c_od=10.0, c_r=7.0, D_tot=80.0):
    steps = len(c_sp_trace)
    env = RollingEnvironment(n_devices=3, max_queue=100.0)
    
    alpha_prev = 0.0
    total_u0 = 0.0
    total_cost = 0.0
    alpha_hist = []
    
    # State variables for the reactive baseline
    reactive_cooldown = 0
    alpha_react = 0.8
    
    for t in range(steps):
        c_sp = c_sp_trace[t]
        eps = eps_trace[t]
        w_t = env.get_w_t()
        
        # --- POLICY LOGIC ---
        if policy_name == "Stackelberg":
            rho_t = 0.30
            eta_t = eps * ((1.0 - rho_t) ** 2)
            alpha_t, step_u0 = solve_stackelberg(
                env.theta, w_t, eta_t, env.r_bar, c_od, c_sp, c_r, rho_t, D_tot,
                alpha_prev=alpha_prev, delta=0.5
            )
        elif policy_name == "All_OnDemand":
            alpha_t, rho_t = 0.0, 0.0
        elif policy_name == "All_Spot":
            alpha_t, rho_t = 1.0, 0.0
        elif policy_name == "Static_50":
            alpha_t, rho_t = 0.50, 0.20
        elif policy_name == "Reactive":
            # Reactive heuristic: If queues are filling up (>50%), panic and drop spot usage.
            # Otherwise, slowly increase spot usage to save money.
            rho_t = 0.10
            queue_pressure = np.max(env.Q / env.Q_max)
            
            if queue_pressure > 0.5 or reactive_cooldown > 0:
                alpha_t = 0.10  # Fallback to mostly On-Demand
                reactive_cooldown = max(0, reactive_cooldown - 1)
                if queue_pressure > 0.8:
                    reactive_cooldown = 3  # Stay on demand if critically full
            else:
                alpha_t = min(0.90, alpha_react + 0.05)
                
            alpha_react = alpha_t

        # --- EXECUTE IN ENVIRONMENT ---
        # Calculate utility for the baselines (Stackelberg already returns this)
        if policy_name != "Stackelberg":
            eta_t = eps * ((1.0 - rho_t) ** 2)
            step_u0 = leader_utility(alpha_t, env.theta, w_t, eta_t, env.r_bar, c_od, c_sp, c_r, rho_t, D_tot, alpha_prev, delta=0.5)
            
        _, r_transmitted, dropped = env.step(alpha_t, rho_t, eps, c_sp, c_od, c_r, D_tot)
        
        fleet_cost = (1 - alpha_t) * c_od + alpha_t * c_sp + rho_t * alpha_t * c_r
        
        total_cost += fleet_cost
        total_u0 += step_u0
        alpha_prev = alpha_t
        alpha_hist.append(alpha_t)
        
    return {
        "Utility": total_u0,
        "Cost": total_cost,
        "Spot_Pct": np.mean(alpha_hist) * 100,
        "Drops": env.total_dropped
    }

def main():
    print("Fetching AWS Trace for Baseline Comparison...")
    c_sp_trace, eps_trace = fetch_web_spot_trace(region="us-east-1", instance_type="c5.large", T=720)
    
    policies = ["All_OnDemand", "All_Spot", "Static_50", "Reactive", "Stackelberg"]
    results = {}
    
    print("\nSimulating 720 hours across all policies...\n")
    for p in policies:
        results[p] = run_policy(p, c_sp_trace, eps_trace)
        
    print(f"{'Policy Name':<15} | {'Net Utility':<12} | {'Fleet Cost':<12} | {'Avg Spot %':<10} | {'SLA Drops (Packets)'}")
    print("-" * 75)
    for p in policies:
        res = results[p]
        print(f"{p:<15} | ${res['Utility']:<11.2f} | ${res['Cost']:<11.2f} | {res['Spot_Pct']:>6.1f}%   | {res['Drops']:.0f}")

if __name__ == "__main__":
    main()