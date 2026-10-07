from pydantic import BaseModel, Field
from typing import List

class OptimizeRequest(BaseModel):
    theta: List[float] = Field(..., description="Latency sensitivity weights for devices")
    w_t: List[float] = Field(..., description="Effective SLA risk (omega_tilde) of devices")
    eta: float = Field(..., description="Effective cloud eviction risk")
    r_bar: List[float] = Field(..., description="Maximum physical sending caps of devices")
    
    c_od: float = Field(..., gt=0, description="On-demand price")
    c_sp: float = Field(..., gt=0, description="Spot instance price")
    c_r: float = Field(..., gt=0, description="Cost of reserve capacity")
    rho: float = Field(default=0.30, ge=0.0, le=1.0, description="Reserve capacity ratio")
    
    D_tot: float = Field(..., description="Total system demand multiplier")
    alpha_prev: float = Field(default=0.0, ge=0.0, le=1.0, description="Previous allocation state")
    delta: float = Field(default=0.0, description="Stateful migration/switching penalty")

class OptimizeResponse(BaseModel):
    alpha_star: float
    follower_rate_total: float
    cloud_cost: float
    status: str