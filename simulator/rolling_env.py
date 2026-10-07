import numpy as np
from core.solver import leader_utility
from core.follower import follower_rate, mitigation


class RollingEnvironment:
    """Stateful edge-device queue simulator for the Stackelberg control loop.

    Parameters
    ----------
    n_devices : int
        Number of IoT/CPS edge devices.
    max_queue : float
        Maximum buffer capacity per device (packets).
    theta, base_omega, r_bar, arrival_rate : np.ndarray or None
        Per-device parameters.  When None, use the 3-device defaults
        from the proposal's worked example.
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

        self.total_dropped = 0.0

        # Per-instance RNG for reproducibility
        self._rng = np.random.default_rng(seed)

    def get_w_t(self):
        """Map queue backlog to SLA risk weight (w_t). Fuller queue = higher panic."""
        queue_fullness = self.Q / self.Q_max
        return self.base_omega * (1.0 + 3.0 * queue_fullness)

    def step(self, alpha, rho, eps_true, c_sp, c_od, c_r, D_tot):
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

        # 1. Devices determine transmission rate based on announced risk
        r_transmitted = follower_rate(alpha, w_t, eta_true, self.r_bar)

        # 2. Cloud Provider Utility
        u0 = leader_utility(alpha, self.theta, w_t, eta_true, self.r_bar,
                            c_od, c_sp, c_r, rho, D_tot)

        # 3. Physical Eviction Realization (per-instance RNG)
        if self._rng.random() < eps_true:
            surviving_capacity = (1.0 - alpha) + (rho * alpha)
            r_transmitted = r_transmitted * surviving_capacity

        # 4. True Queue Update
        self.Q = self.Q + self.arrival_rate - r_transmitted

        # 5. Track Overflows (SLA Violations / Packet Drops)
        overflows = np.maximum(0.0, self.Q - self.Q_max)
        self.total_dropped += float(np.sum(overflows))

        # Cap queues at max capacity for the next step
        self.Q = np.clip(self.Q, 0.0, self.Q_max)

        return u0, r_transmitted, float(np.sum(overflows))