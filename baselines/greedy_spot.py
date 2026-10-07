from core.solver import leader_utility

def solve_greedy_spot(theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot):
    """Baseline 2: 100% Spot instances. Minimum base cost, maximum eviction penalty risk."""
    alpha = 1.0
    # Assumes no reserve mitigation is used in a purely greedy approach
    u = leader_utility(alpha, theta, w_t, eta, r_bar, c_od, c_sp, c_r, 0.0, D_tot)
    return alpha, u