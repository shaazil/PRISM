from core.solver import leader_utility

def solve_static_split(theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot, fixed_alpha=0.5):
    """Baseline 3: Fixed split between Spot and On-Demand."""
    u = leader_utility(fixed_alpha, theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot)
    return fixed_alpha, u