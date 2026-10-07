from core.solver import leader_utility

def solve_on_demand(theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot):
    """Baseline 1: 100% On-Demand instances. Zero eviction risk, max cost."""
    alpha = 0.0
    # Reserve ratio (rho) is irrelevant when alpha is 0
    u = leader_utility(alpha, theta, w_t, eta, r_bar, c_od, c_sp, c_r, 0.0, D_tot)
    return alpha, u