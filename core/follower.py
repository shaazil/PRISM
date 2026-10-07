import numpy as np

def mitigation(rho):
    """Diminishing-returns reserve mitigation g(rho) = (1 - rho)^2, so eta = eps * g(rho)."""
    return (1.0 - rho) ** 2

def eff_omega(omega, H, tau):
    """Calculates effective risk sensitivity based on buffer horizon."""
    return omega * np.exp(-H / tau)

def follower_rate(alpha, w_t, eta, r_bar):
    """
    Follower best response: how fast IoT devices send data.
    alpha: Spot fraction announced by leader
    w_t: Effective SLA risk (omega_tilde) of the devices
    eta: Effective cloud eviction risk
    r_bar: Maximum physical sending caps of the devices
    """
    nu = w_t * eta * alpha                 # 'risk price' per unit rate
    
    # If risk is > 0, compute optimal rate. Otherwise, send at max physical cap.
    # The 1e-300 prevents division by zero if alpha or eta are exactly 0.
    r = np.where(nu > 0, 1.0 / np.maximum(nu, 1e-300) - 1.0, r_bar)
    
    # Clip the results so devices don't send negative data or exceed their hardware cap
    return np.clip(r, 0.0, r_bar)