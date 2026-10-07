import numpy as np

def calculate_cost_savings(stackelberg_cost, baseline_cost):
    """
    Computes percentage cost reduction vs. a baseline (usually Pure On-Demand).
    Target stat for resume: "Reduced procurement costs by X%"
    """
    if baseline_cost <= 0:
        return 0.0
    return ((baseline_cost - stackelberg_cost) / baseline_cost) * 100.0

def calculate_sla_violations(delivered_rates, required_rates, tolerance=0.90):
    """
    Measures how often the system fails to meet device throughput needs due to spot evictions.
    Target stat for resume: "Maintained 99.X% SLA compliance during eviction spikes."
    """
    # A violation occurs if delivered rate drops below 90% of the required rate
    violations = np.sum(delivered_rates < (required_rates * tolerance))
    total_intervals = len(delivered_rates)
    
    violation_rate = (violations / total_intervals) * 100.0
    return violation_rate

def system_efficiency_score(cost_savings_pct, violation_rate_pct, penalty_weight=2.0):
    """
    A custom composite metric to evaluate the trade-off. 
    Rewards cost savings but heavily penalizes SLA violations.
    """
    return cost_savings_pct - (violation_rate_pct * penalty_weight)