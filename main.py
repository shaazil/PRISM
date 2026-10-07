import numpy as np
from simulator.trace_loader import fetch_web_spot_trace
from simulator.rolling_env import RollingEnvironment
from core.solver import solve_stackelberg

def run_dynamic_simulation():
    print("=== Initializing Dynamic Stackelberg Cloud Controller ===")
    
    # 1. Fetch real pricing and risk data directly from the public AWS web API
    c_sp_trace, eps_trace = fetch_web_spot_trace(region="us-east-1", instance_type="c5.large", c_od=10.0, T=720)
    
    # 2. Initialize the stateful edge environment (true queue dynamics)
    n_devices = 3
    env = RollingEnvironment(n_devices=n_devices, max_queue=100.0)
    
    # Global system parameters
    c_od = 10.0           # On-Demand baseline price
    c_r = 7.0             # Reserved instance price
    rho_fixed = 0.30      # 30% warm reserve buffer
    D_tot = 80.0          # Total SLA penalty scale
    delta = 0.5           # Migration penalty (cost to switch instances)
    
    # State tracking
    alpha_prev = 0.0
    total_leader_utility = 0.0
    alpha_history = []
    
    total_steps = len(c_sp_trace)
    print(f"\nStarting {total_steps}-step rolling horizon simulation...")
    
    # 3. The true rolling-horizon dynamic loop
    for t in range(total_steps):
        c_sp_t = c_sp_trace[t]
        eps_t = eps_trace[t]
        
        # A. Environment provides real-time SLA panic weights based on physical queues
        w_t = env.get_w_t()
        
        # B. Leader computes optimal spot fraction, penalizing drastic shifts from alpha_prev
        eta_t = eps_t * ((1.0 - rho_fixed) ** 2) 
        alpha_star, step_u0 = solve_stackelberg(
            theta=env.theta, w_t=w_t, eta=eta_t, r_bar=env.r_bar, 
            c_od=c_od, c_sp=c_sp_t, c_r=c_r, rho=rho_fixed, D_tot=D_tot, 
            alpha_prev=alpha_prev, delta=delta
        )
        
        # C. Step the physical environment forward (data generated, transmitted, or dropped)
        step_u0, r_transmitted, dropped_packets = env.step(
            alpha=alpha_star, rho=rho_fixed, eps_true=eps_t, 
            c_sp=c_sp_t, c_od=c_od, c_r=c_r, D_tot=D_tot
        )
        
        # Update states for next minute
        alpha_prev = alpha_star
        total_leader_utility += step_u0
        alpha_history.append(alpha_star)

    # 4. Final Output Metrics
    avg_alpha = np.mean(alpha_history)
    print("\n=== Simulation Complete ===")
    print(f"Total Leader Utility (Revenue - Costs):  ${total_leader_utility:,.2f}")
    print(f"Average Spot Instance Fraction:          {avg_alpha * 100:.1f}%")
    print(f"Total Packets Dropped (SLA Violations):  {env.total_dropped:.0f}")

if __name__ == "__main__":
    run_dynamic_simulation()