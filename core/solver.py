import numpy as np
from .follower import follower_rate

def leader_utility(alpha, theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot, alpha_prev=0.0, delta=0.0):
    """Exact U0(alpha) with the followers best-responding (caps included) + switching penalty."""
    r = follower_rate(alpha, w_t, eta, r_bar)
    revenue = np.sum(theta * np.log1p(r))
    
    # Standard fleet and SLA costs
    cost = (1 - alpha) * c_od + alpha * c_sp + rho * alpha * c_r
    payout = D_tot * eta * alpha ** 2
    
    # Stateful migration penalty (cost to provision/deprovision VMs)
    switching_penalty = delta * np.abs(alpha - alpha_prev)
    
    return revenue - cost - payout - switching_penalty

def solve_stackelberg(theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot, alpha_prev=0.0, delta=0.0, a_max=1.0):
    """Global leader optimum via the finite candidate set, now including migration kinks."""
    S = c_od - c_sp - rho * c_r                 
    m = D_tot * eta
    lo = 1.0 / (w_t * eta * (1 + r_bar))        
    hi = 1.0 / (w_t * eta)                      
    
    # add alpha_prev to the candidate breakpoints because the absolute 
    # value in the switching penalty creates a mathematical "kink" exactly at this point.
    bps_list = [0.0, a_max, alpha_prev]
    bps = np.unique(np.clip(np.concatenate([bps_list, lo, hi]), 0.0, a_max))
    cands = list(bps)
    
    for a, b in zip(bps[:-1], bps[1:]):
        mid = 0.5 * (a + b)
        theta_seg = theta[(mid > lo) & (mid < hi)].sum()   
        
        # The derivative shifts based on whether we are scaling UP or DOWN from previous state
        S_eff = S - delta if mid > alpha_prev else S + delta

        if theta_seg == 0:
            cands.append(np.clip(S_eff / (2 * m), a, b))       
        else:
            disc = S_eff ** 2 - 8 * m * theta_seg
            if disc >= 0:
                for sign in (+1, -1):
                    root = (S_eff + sign * np.sqrt(disc)) / (4 * m)
                    if a <= root <= b:
                        cands.append(root)
                        
    cands = [c for c in cands if 0.0 <= c <= a_max]
    vals = [leader_utility(c, theta, w_t, eta, r_bar, c_od, c_sp, c_r, rho, D_tot, alpha_prev, delta) for c in cands]
    k = int(np.argmax(vals))
    return cands[k], vals[k]