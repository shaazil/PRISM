import numpy as np
from core.solver import leader_utility
from core.follower import follower_rate, mitigation

class RollingEnvironment:
    def __init__(self, n_devices=3, max_queue=100.0):
        self.n = n_devices
        # Q_i(t): True data backlog for each device
        self.Q = np.zeros(self.n)          
        self.Q_max = np.full(self.n, max_queue)
        
        # Device & SLA parameters
        self.theta = np.array([0.4, 0.3, 0.3])
        self.base_omega = np.array([12.0, 13.0, 9.0])
        self.r_bar = np.full(self.n, 10.0) # Max physical transmission rate
        self.arrival_rate = np.array([4.0, 3.5, 5.0]) # Continuous data generated per step
        
        self.total_dropped = 0.0

    def get_w_t(self):
        """Map queue backlog to SLA risk weight (w_t). Fuller queue = higher panic."""
        # As the queue fills up, the device becomes exponentially more risk-averse
        queue_fullness = self.Q / self.Q_max
        return self.base_omega * (1.0 + 3.0 * queue_fullness)

    def step(self, alpha, rho, eps_true, c_sp, c_od, c_r, D_tot):
        """Executes one real-time step of the dynamic system."""
        w_t = self.get_w_t()
        eta_true = eps_true * mitigation(rho)
        
        # 1. Devices determine transmission rate based on announced risk
        r_transmitted = follower_rate(alpha, w_t, eta_true, self.r_bar)
        
        # 2. Cloud Provider Utility (Before switching costs)
        u0 = leader_utility(alpha, self.theta, w_t, eta_true, self.r_bar, c_od, c_sp, c_r, rho, D_tot)
        
        # 3. Physical Eviction Realization
        evicted = np.random.random() < eps_true
        if evicted:
            # Outflow is bottlenecked by surviving capacity: On-Demand + Warm Reserve
            surviving_capacity = (1.0 - alpha) + (rho * alpha)
            r_transmitted *= surviving_capacity
            
        # 4. True Queue Update
        self.Q = self.Q + self.arrival_rate - r_transmitted
        
        # 5. Track Overflows (SLA Violations / Packet Drops)
        overflows = np.maximum(0.0, self.Q - self.Q_max)
        self.total_dropped += np.sum(overflows)
        
        # Cap queues at max capacity for the next step
        self.Q = np.clip(self.Q, 0.0, self.Q_max)
        
        return u0, r_transmitted, np.sum(overflows)