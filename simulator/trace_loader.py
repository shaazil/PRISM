import urllib.request
import json
import numpy as np

# Official public AWS bucket (No authentication required)
AWS_SPOT_ADVISOR_URL = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"

# Map AWS interruption brackets to eviction risk (eps)
BRACKET_TO_EPS = {0: 0.025, 1: 0.055, 2: 0.090, 3: 0.135, 4: 0.185}

def fetch_web_spot_trace(region="ap-south-1", instance_type="c5.large", c_od=10.0, T=720):
    """
    Fetches real live Spot pricing data from the public AWS S3 JSON endpoint,
    bypassing the need for an authenticated AWS account.
    """
    print(f"Fetching real AWS Spot data from the web for {instance_type} in {region}...")
    
    req = urllib.request.Request(AWS_SPOT_ADVISOR_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        
    try:
        # Extract live data for the specific instance and region
        reg_data = data["spot_advisor"][region]["Linux"][instance_type]
        savings_pct = reg_data["s"]
        r_idx = reg_data["r"]
        
        # Calculate real baseline price and risk
        real_c_sp_base = c_od * (1.0 - savings_pct / 100.0)
        real_eps_base = BRACKET_TO_EPS.get(r_idx, 0.06)
        
        print(f"Live Data Found -> Savings: {savings_pct}%, Base Spot Price: ${real_c_sp_base:.2f}, Risk Bracket: {r_idx}")
        
    except KeyError:
        print(f"Warning: {instance_type} not found in {region}. Falling back to default estimates.")
        real_c_sp_base, real_eps_base = 3.5, 0.06

    # Synthesize a dynamic time-series around the real baseline pulled from the web
    rng = np.random.default_rng(42)
    t = np.arange(T, dtype=float)
    
    # Add diurnal waves and noise to the real base price
    diurnal = 0.45 * np.sin(2.0 * np.pi * t / 24.0)
    c_sp_trace = np.clip(real_c_sp_base + diurnal + rng.normal(0, 0.15, size=T), 1.0, c_od)
    
    # Scale risk dynamically as the price fluctuates
    eps_trace = np.clip(real_eps_base + 0.01 * diurnal + rng.normal(0, 0.005, size=T), 0.01, 0.25)
    
    return c_sp_trace, eps_trace