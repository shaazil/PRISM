from fastapi import FastAPI, HTTPException, Response
from api.schemas import OptimizeRequest, OptimizeResponse
from core.solver import solve_stackelberg
from core.follower import follower_rate
import numpy as np
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="PRISM API",
    description="Stackelberg Game-Theoretic Cloud Elasticity Optimizer",
    version="1.0.0"
)

# --- Prometheus Metrics Definitions ---
OPTIMIZATION_CALLS = Counter("prism_optimization_calls_total", "Total times the optimizer was called")
CURRENT_ALPHA = Gauge("prism_alpha_star", "Current optimal spot allocation fraction")
CURRENT_COST = Gauge("prism_cloud_cost", "Current computed cloud cost")
TOTAL_FOLLOWER_RATE = Gauge("prism_follower_rate", "Total follower data rate")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "PRISM-Engine"}

@app.get("/metrics")
def get_metrics():
    """Endpoint for Prometheus to scrape metrics."""
    # generate_latest() compiles all our Counters and Gauges into the format Prometheus expects
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/optimize", response_model=OptimizeResponse)
def optimize_allocation(req: OptimizeRequest):
    try:
        OPTIMIZATION_CALLS.inc()  # Increment counter on every request
        
        theta_arr = np.array(req.theta)
        w_t_arr = np.array(req.w_t)
        r_bar_arr = np.array(req.r_bar)
        
        alpha_star, _ = solve_stackelberg(
            theta=theta_arr,
            w_t=w_t_arr,
            eta=req.eta,
            r_bar=r_bar_arr,
            c_od=req.c_od,
            c_sp=req.c_sp,
            c_r=req.c_r,
            rho=req.rho,
            D_tot=req.D_tot,
            alpha_prev=req.alpha_prev,
            delta=req.delta
        )
        
        follower_rates = follower_rate(
            alpha=alpha_star,
            w_t=w_t_arr,
            eta=req.eta,
            r_bar=r_bar_arr
        )
        
        total_rate = float(np.sum(follower_rates))
        effective_price = (1 - alpha_star) * req.c_od + alpha_star * req.c_sp + req.rho * alpha_star * req.c_r
        cost = total_rate * effective_price

        # Update Gauges with the latest computed values
        CURRENT_ALPHA.set(float(alpha_star))
        CURRENT_COST.set(float(cost))
        TOTAL_FOLLOWER_RATE.set(total_rate)

        return OptimizeResponse(
            alpha_star=round(float(alpha_star), 4),
            follower_rate_total=round(total_rate, 4),
            cloud_cost=round(float(cost), 4),
            status="optimal"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))