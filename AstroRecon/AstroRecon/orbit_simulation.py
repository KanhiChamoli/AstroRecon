import math
import csv

def run_simulation():
    # Physical constants and parameters for Sagittarius A*
    G = 4.3e-3       # Gravitational constant
    mass_bh = 4.0e6  # Mass of Sagittarius A* (4 million solar masses)
    
    # Initial position of the star (parsecs)
    x = 0.1
    y = 0.0
    
    # Initial velocity of the star
    vx = 0.0
    vy = 1000.0
    
    dt = 0.001       # Time step
    steps = 1000     # Total steps
    
    print("Running Sagittarius A* orbital simulation...")
    
    # Open a CSV file to record trajectory data
    with open('orbit_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'X', 'Y'])
        
        for step in range(steps):
            # Calculate radial distance r using Pythagorean theorem
            r = math.sqrt(x*x + y*y)
            if r < 0.001:
                break
                
            # Compute gravitational acceleration components
            accel_factor = (G * mass_bh) / (r * r * r)
            ax = -accel_factor * x
            ay = -accel_factor * y
            
            # Update velocity
            vx += ax * dt
            vy += ay * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            
            # Write data row
            writer.writerow([step, x, y])
            
    print("Simulation complete! Data successfully saved to 'orbit_data.csv'.")

if __name__ == "__main__":
    run_simulation()