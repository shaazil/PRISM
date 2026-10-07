import requests
import time
import random
import math

API_URL = "http://localhost:8000/optimize"

print("Starting PRISM Live Traffic Simulator...")
print("Press Ctrl+C to stop.")

step = 0
try:
    while True:
        # Simulate a fluctuating spot price using a sine wave + some random noise
        # Base price $0.05, fluctuating between $0.02 and $0.08
        simulated_spot_price = 0.05 + (0.03 * math.sin(step / 10.0)) + random.uniform(-0.005, 0.005)
        simulated_spot_price = max(0.01, round(simulated_spot_price, 4))
        
        # Simulate fluctuating device arrival rates
        base_arrival = random.uniform(4.0, 8.0)
        
        payload = {
            "theta": [0.4, 0.3, 0.3],
            "w_t": [1.5, 2.0, 1.2],
            "eta": 0.05,
            "r_bar": [10.0, 12.0, 8.0],
            "c_od": 0.10,
            "c_sp": simulated_spot_price,
            "c_r": 0.05,
            "rho": 0.30,
            "D_tot": 100.0,
            "alpha_prev": 0.0,
            "delta": 0.01
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            data = response.json()
            print(f"Step {step} | Spot Price: ${simulated_spot_price:.4f} | Alpha*: {data['alpha_star']} | Cost: ${data['cloud_cost']}")
        except requests.exceptions.ConnectionError:
            print("Failed to connect to API. Is Docker running?")
            
        step += 1
        time.sleep(2) # Wait 2 seconds before the next request

except KeyboardInterrupt:
    print("\nSimulation stopped.")